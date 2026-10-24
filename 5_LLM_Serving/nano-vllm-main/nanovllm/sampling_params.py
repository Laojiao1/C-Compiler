from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0    # 温度，控制采样随机性
    max_tokens: int = 64        # 最多生成多少个 token
    ignore_eos: bool = False    # 是否忽略 EOS 继续生成，主要用于压力测试，都跑满才能比较吞吐数据

    def __post_init__(self):
        # 断言温度必须大于 1e-10，其实是一个简化：它的 Sampler 走的是统一的随机采样路径，
        # 没有为 temperature=0 单独写一条 argmax 分支
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
