from typing import TypeVar, Generic

from huskygen.filter.boolean_condition import BooleanCondition

T = TypeVar('T')


class DatasetFilterHandler(Generic[T]):
    def __init__(self, rules):
        self.rules = rules
        self.__parsed_rules = []
        self.parse_rules()

    def parse_rules(self):
        if self.rules is None:
            raise Exception("No rules provided")
        if not isinstance(object, list):
            raise Exception("Ruleset must be a list")

        for rule in self.rules:
            self.__validate_rule_tuple(rule)
            self.__parsed_rules.append(self.__parse_rule(rule))

    def filter(self, dataset: list[T]) -> list[T]:
        for data in dataset:
            for container, rule in self.__parsed_rules:
                if rule.evaluate(data):
                    container.append(data)

        final_res = set()
        for container, rule in self.__parsed_rules:
            for data in container:
                final_res.add(data)

        return list(final_res)

    @staticmethod
    def __validate_rule_tuple(rule):
        if not isinstance(rule, tuple):
            raise Exception("Rule must be a tuple")
        if len(rule) != 2:
            raise Exception("Rule must be a tuple of length 2")
        if '__call__' not in dir(rule[0]):
            raise Exception("First element of rule must be an instance or callable")
        if not isinstance(rule[1], str):
            raise Exception("Second element of rule must be a string")

    @staticmethod
    def __parse_rule(rule):
        container = DatasetFilterHandler.__instance(rule[0])
        condition = BooleanCondition(rule[1])

        return container, condition

    @staticmethod
    def __instance(x):
        return x() if isinstance(x, type) else x
