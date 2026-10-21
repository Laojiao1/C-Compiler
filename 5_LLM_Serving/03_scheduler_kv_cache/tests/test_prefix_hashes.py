"""用内容关系与跨进程稳定性验证前缀指纹，不涉及真实模型分词。"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from prefix_cache import build_prefix_hashes


class PrefixHashTests(unittest.TestCase):
    def test_only_complete_blocks_are_hashed(self):
        namespace = "demo-model:revision-1:fp32"
        self.assertEqual(build_prefix_hashes([], 4, namespace), [])
        self.assertEqual(build_prefix_hashes([1, 2, 3], 4, namespace), [])
        tokens = list(range(10))
        original = tokens.copy()
        hashes = build_prefix_hashes(tokens, 4, namespace)
        self.assertEqual(len(hashes), 2)
        self.assertEqual(tokens, original)
        # 改动未满尾块不应改变已经完整的两块指纹。
        changed_tail = list(range(8)) + [90, 91]
        self.assertEqual(build_prefix_hashes(changed_tail, 4, namespace), hashes)
        extended = build_prefix_hashes(list(range(12)), 4, namespace)
        self.assertEqual(len(extended), 3)
        self.assertEqual(extended[:2], hashes)

    def test_parent_context_changes_later_hashes(self):
        first = [1, 2, 3, 4, 10, 11, 12, 13]
        second = [9, 2, 3, 4, 10, 11, 12, 13]
        hashes_a = build_prefix_hashes(first, 4, "demo:fp32")
        hashes_b = build_prefix_hashes(second, 4, "demo:fp32")
        self.assertNotEqual(hashes_a[0], hashes_b[0])
        # 第二块的局部 Token 相同，但此前上下文不同，不能误认成同一前缀。
        self.assertNotEqual(hashes_a[1], hashes_b[1])

    def test_divergence_propagates_after_common_prefix(self):
        tokens_a = list(range(12))
        tokens_b = tokens_a.copy()
        tokens_b[4] = 99
        hashes_a = build_prefix_hashes(tokens_a, 4, "demo:fp32")
        hashes_b = build_prefix_hashes(tokens_b, 4, "demo:fp32")
        self.assertEqual(hashes_a[0], hashes_b[0])
        self.assertNotEqual(hashes_a[1], hashes_b[1])
        # 第三块的局部 Token 又相同，但已经发生分叉，后续指纹仍必须不同。
        self.assertNotEqual(hashes_a[2], hashes_b[2])

    def test_namespace_separates_computation_settings(self):
        tokens = list(range(8))
        hashes_a = build_prefix_hashes(tokens, 4, "demo:revision-1:fp32")
        for namespace in ("demo:revision-2:fp32", "demo:revision-1:bf16"):
            with self.subTest(namespace=namespace):
                hashes_b = build_prefix_hashes(tokens, 4, namespace)
                self.assertTrue(all(a != b for a, b in zip(hashes_a, hashes_b)))

    def test_hashes_are_stable_across_processes(self):
        # 改变 Python 内置 hash 的随机种子，检验实现是否仍有稳定输出。
        script = (
            "import json; from prefix_cache import build_prefix_hashes; "
            "print(json.dumps(build_prefix_hashes(list(range(12)), 4, 'demo:fp32')))"
        )
        project_dir = Path(__file__).resolve().parents[1]
        results = []
        for seed in ("11", "22"):
            child_env = os.environ.copy()
            child_env["PYTHONHASHSEED"] = seed
            output = subprocess.check_output(
                [sys.executable, "-B", "-c", script],
                cwd=project_dir,
                env=child_env,
                text=True,
                timeout=20,
            )
            results.append(json.loads(output))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(results[0]), 3)
        for digest in results[0]:
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_invalid_inputs(self):
        for tokens, block_size, namespace in [
            ([1, 2], 0, "demo"),
            ([1, 2], -4, "demo"),
            ([1, 2], 4, ""),
            ([1, -2], 4, "demo"),
            ([1, 2.5], 4, "demo"),
            ([1, True], 4, "demo"),
        ]:
            with self.subTest(tokens=tokens, block_size=block_size, namespace=namespace):
                with self.assertRaises(ValueError):
                    build_prefix_hashes(tokens, block_size, namespace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
