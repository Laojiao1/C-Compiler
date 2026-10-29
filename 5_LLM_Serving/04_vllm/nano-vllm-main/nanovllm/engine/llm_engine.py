# 引擎主循环：调度 → 前向 → 采样 → 更新
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 1. 从 kwargs 里挑出 Config 认识的字段，构造全局配置
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 把块大小同步给 Sequence（它用类变量记 block_size）
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []

        # 2. 张量并行：用 spawn 启动 rank > 0 的 worker 子进程
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 的 ModelRunner 跑在主进程
        self.model_runner = ModelRunner(config, 0, self.events)

        # 3. 分词器只在主进程持有，并把 EOS 回填到 config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 4. 调度器 + 退出清理
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # add_request 把一条 prompt 变成一个 Sequence 放进调度器。
    # prompt 可以是字符串（内部先 encode）也可以是现成的 token id 列表：
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 如果传过来的是字符串
        if isinstance(prompt, str):
            # 调用 HuggingFace 分词器，将字符串转换为整数列表
            prompt = self.tokenizer.encode(prompt)
        # 如果传入的就是整数列表，跳过 if，构造 Sequence 对象
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 一次完整的 调度 -> 前向 -> 写回
    def step(self):
        # 1. 调度：选出 Batch 并分配显存块
        seqs, is_prefill = self.scheduler.schedule() # 选 batch

        # 注意这里的符号编码：+表示 prefill，-表示 decode

        # 后续 generate 靠符号区分当前是 prefill 还是 decode 阶段
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs) 

        # 2. 跑前向：GPU 前向计算与采样
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 3. 写回：状态更新与显存回收
        self.scheduler.postprocess(seqs, token_ids, is_prefill) 

        # outputs 只挑选 is_finished == True 的序列
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # 事件驱动主循环：注册请求 → 驱动系统时钟（step）→ 收集已经完结的请求
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 进度条
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 将 sampling_params 广播成与 prompts 等长的列表，为了后续的批处理实例化
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 1. 遍历输入，调用 add_request() 登记每条请求，批量实例化 Sequence 并挂入 waiting 队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.

        while not self.is_finished():
            t = perf_counter() # perf_counter 是一个专门用来做高精度代码测速的系统级计时器

            # 2. 只要 Scheduler 里还有未完成的请求，就反复调用 step (调度 -> GPU 前向 -> 状态写回)
            output, num_tokens = self.step() # 跑一步，拿回本步完成的序列

            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            for seq_id, token_ids in output:
                # 乱序收集结果：如果有序列刚好跑完 -> outputs[seq_id] = token_ids
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()

        # 3. 按 seq_id 重新排好序 ──> Tokenizer 解码文本 ──> 返回给用户
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
 