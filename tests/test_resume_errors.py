import unittest

from src.pipeline import input_matches_current_schema, record_matches_current_schema


class ResumeErrorTests(unittest.TestCase):
    def test_judge_error_does_not_count_as_done(self):
        self.assertFalse(record_matches_current_schema("judge", {"judge_error": True}))
        self.assertTrue(record_matches_current_schema("judge", {"judge_error": False}))
        self.assertTrue(record_matches_current_schema("judge", {}))

    def test_grade_error_does_not_count_as_done(self):
        self.assertFalse(record_matches_current_schema("grade", {"quality": "error"}))
        self.assertTrue(record_matches_current_schema("grade", {"quality": "type1"}))

    def test_failed_solve_records_do_not_feed_judge(self):
        self.assertFalse(input_matches_current_schema("judge", {"solve_error": True}))
        self.assertTrue(input_matches_current_schema("judge", {"source": "NEW"}))

    def test_failed_judge_records_do_not_feed_grade(self):
        self.assertFalse(input_matches_current_schema("grade", {"judge_error": True}))
        self.assertTrue(input_matches_current_schema("grade", {"result": "new"}))


if __name__ == "__main__":
    unittest.main()
