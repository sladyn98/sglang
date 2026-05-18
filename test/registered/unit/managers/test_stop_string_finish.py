import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStopStringFinish(unittest.TestCase):
    def _new_req(self, decoded_text: str, tail_str: str, stop_strs: list[str]) -> Req:
        req = Req.__new__(Req)
        req.decoded_text = decoded_text
        req.finished_reason = None
        req.sampling_params = SimpleNamespace(
            stop_strs=stop_strs,
            stop_regex_strs=[],
        )
        req.tail_str = lambda: tail_str
        return req

    def test_uses_earliest_stop_string_in_text_order(self):
        req = self._new_req(
            decoded_text="",
            tail_str="answer <early-stop> middle <late-stop>",
            stop_strs=["<late-stop>", "<early-stop>"],
        )

        self.assertTrue(req._check_str_based_finish())
        self.assertEqual(req.finished_reason.to_json()["matched"], "<early-stop>")

    def test_uses_earliest_stop_string_across_decoded_text_and_tail(self):
        req = self._new_req(
            decoded_text="answer <early-stop>",
            tail_str="<early-stop> middle <late-stop>",
            stop_strs=["<late-stop>", "<early-stop>"],
        )

        self.assertTrue(req._check_str_based_finish())
        self.assertEqual(req.finished_reason.to_json()["matched"], "<early-stop>")

    def test_no_match_does_not_finish(self):
        req = self._new_req(
            decoded_text="answer",
            tail_str=" still running",
            stop_strs=["<stop>"],
        )

        self.assertFalse(req._check_str_based_finish())
        self.assertIsNone(req.finished_reason)


if __name__ == "__main__":
    unittest.main()
