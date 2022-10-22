import unittest

from huskygen.filter.boolean_condition import BooleanCondition


class BooleanConditionTest(unittest.TestCase):
    def test_numeric_condition(self):
        cond = BooleanCondition("a > 15 && b < 20")
        self.assertTrue(cond.evaluate({"a": 16, "b": 19}))
        self.assertFalse(cond.evaluate({"a": 16, "b": 21}))
        self.assertFalse(cond.evaluate({"a": 15, "b": 20}))

    def test_string_condition(self):
        cond = BooleanCondition('a == "abc" && b == "def"')
        self.assertTrue(cond.evaluate({"a": "abc", "b": "def"}))
        self.assertFalse(cond.evaluate({"a": "abc", "b": "defg"}))

    def test_variable_condition(self):
        cond = BooleanCondition("a == b")
        self.assertTrue(cond.evaluate({"a": "abc", "b": "abc"}))
        self.assertFalse(cond.evaluate({"a": "abc", "b": "def"}))

    def test_var_or_condition(self):
        cond = BooleanCondition("a == b || c == d")
        self.assertTrue(cond.evaluate({"a": "abc", "b": "abc", "c": "def", "d": "def"}))
        self.assertTrue(cond.evaluate({"a": "abc", "b": "abc", "c": "abc", "d": "def"}))
        self.assertFalse(cond.evaluate({"a": "abc", "b": "def", "c": "def", "d": "dek"}))

    def test_var_and_condition(self):
        cond = BooleanCondition("a == b && c == d")
        self.assertTrue(cond.evaluate({"a": "abc", "b": "abc", "c": "def", "d": "def"}))
        self.assertFalse(cond.evaluate({"a": "abc", "b": "abc", "c": "abc", "d": "def"}))
        self.assertFalse(cond.evaluate({"a": "abc", "b": "def", "c": "def", "d": "dek"}))

    def test_in_condition(self):
        # TODO: Maybe assume that left side is always string?
        cond = BooleanCondition("'abc' IN b")
        self.assertTrue(cond.evaluate({"b": ["abc", "def"]}))
        self.assertFalse(cond.evaluate({"b": ["def", "ghi"]}))

        cond = BooleanCondition("b:abc")
        self.assertTrue(cond.evaluate({"b": ["abc", "def"]}))
        self.assertFalse(cond.evaluate({"b": ["def", "ghi"]}))

        cond = BooleanCondition("b:abc && c:ghi")
        self.assertTrue(cond.evaluate({"b": ["abc", "def"], "c": ["ghi", "jkl"]}))
        self.assertFalse(cond.evaluate({"b": ["abc", "def"], "c": ["jkl", "mno"]}))

        cond = BooleanCondition("b:abc || c:ghi")
        self.assertTrue(cond.evaluate({"b": ["rere", "rere"], "c": ["ds", "ghi"]}))
        self.assertFalse(cond.evaluate({"b": ["123", "456"], "c": ["123", "456"]}))

        cond = BooleanCondition("123:abc || 123:ghi")
        self.assertFalse(cond.evaluate({"b": ["rere", "rere"], "c": ["ds", "ghi"]}))

    def test_compound_condition(self):
        cond = BooleanCondition("f > -1 || ('abc' IN b && (a == 10 || c == 20) && d <= 1000)")
        self.assertTrue(cond.evaluate({"a": 10, "b": ["abc", "def"], "c": 20, "f": -1, "d": 1000}))


if __name__ == '__main__':
    unittest.main()
