"""固定长度输入及请求参数测试：不发送请求，不要求真实模型加载。"""

import unittest

from stage_3_serving.prepare_mixed_workload import build_requests, make_exact_prompt


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("不应增加特殊 Token")
        return [ord(character) for character in text]


class MixedWorkloadTests(unittest.TestCase):
    def test_repeat_then_truncate(self):
        self.assertEqual(make_exact_prompt([11, 22, 33], 8),
                         [11, 22, 33, 11, 22, 33, 11, 22])

    def test_target_shorter_than_seed(self):
        self.assertEqual(make_exact_prompt([11, 22, 33], 2), [11, 22])

    def test_exact_multiple_and_single_token_seed(self):
        self.assertEqual(make_exact_prompt([11, 22], 4), [11, 22, 11, 22])
        self.assertEqual(make_exact_prompt([9], 3), [9, 9, 9])

    def test_returns_independent_list_without_modifying_seed(self):
        seed = [11, 22]
        result = make_exact_prompt(seed, 2)
        result[0] = 99
        self.assertEqual(seed, [11, 22])

    def test_invalid_inputs(self):
        for seed, target in [([], 4), ([1], 0), ([1], -1), ([1], True),
                             ([1], 2.5), ([-1], 3), ([True], 3), (["1"], 3)]:
            with self.subTest(seed=seed, target=target), self.assertRaises(ValueError):
                make_exact_prompt(seed, target)

    def test_seven_requests_have_expected_input_and_output_lengths(self):
        requests = build_requests(FakeTokenizer())
        self.assertEqual([r["request_id"] for r in requests],
                         ["short_0", "short_1", "short_2", "short_3",
                          "long_0", "long_1", "long_2"])
        for request in requests:
            expected = (128, 256) if request["cohort"] == "short" else (1536, 32)
            self.assertEqual((len(request["body"]["prompt"]),
                              request["body"]["max_tokens"]), expected)
            self.assertEqual(request["expected_prompt_tokens"], expected[0])
            self.assertEqual(request["expected_completion_tokens"], expected[1])

    def test_completion_body_fixes_sampling_and_adds_no_chat_template(self):
        for request in build_requests(FakeTokenizer()):
            body = request["body"]
            self.assertNotIn("messages", body)
            self.assertIsInstance(body["prompt"], list)
            self.assertFalse(body["add_special_tokens"])
            self.assertTrue(body["ignore_eos"])
            self.assertEqual(body["temperature"], 0.0)
            self.assertEqual(body["n"], 1)
            self.assertTrue(body["stream"])
            self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_model_length_excess_is_rejected_not_truncated(self):
        with self.assertRaises(ValueError):
            build_requests(FakeTokenizer(), max_model_len=1500)

    def test_prompts_are_different_and_independently_owned(self):
        requests = build_requests(FakeTokenizer())
        prompts = [request["body"]["prompt"] for request in requests]
        self.assertEqual(len(set(tuple(prompt) for prompt in prompts)), 7)
        old_second = prompts[1].copy()
        prompts[0][0] = 99999
        self.assertEqual(prompts[1], old_second)


if __name__ == "__main__":
    unittest.main()
