"""
Grammar:

Expression --> AndTerm { OR AndTerm }+
AndTerm --> Condition { AND Condition }+
Condition --> Terminal (>,<,>=,<=,==) Terminal | (Expression) | AttributeCondition
MembershipCondition -> Key:Value
Terminal --> Number, String, Variable or MembershipCondition

MembershipCondition - Searches for `Key` in the provided objects, and if found, checks if `Value` is in the result if the Value is a list, or if the result is equal to the Value if the Value is a string.

Usage:
p = BooleanCondition('<expression text>')
p.evaluate(variable_dict) # variable_dict is a dictionary providing values for variables that appear in <expression text>
"""
import operator
from enum import Enum
from typing import Optional


class TokenClass(Enum):
    OPERATOR = 1
    TERMINAL = 2
    PARENTHESIS = 3


class Token:
    def __init__(self, cls, rep, func=None):
        self.type = cls
        self.representation = rep
        self.func = func

    def __repr__(self):
        return f"Token({self.type}, {self.representation})"


def evaluate_attribute_token(left, right):
    if isinstance(right, str):
        return left.lower().find(right.lower()) > -1
    elif isinstance(right, list):
        return any(left.lower() == str(elem).lower() for elem in right)

    return False


TOKENS = {
    "NUM": Token(TokenClass.TERMINAL, "[NUMBER]"),
    "STR": Token(TokenClass.TERMINAL, "[LITERAL]"),
    "VAR": Token(TokenClass.TERMINAL, "[VARIABLE]"),
    "BOOL": Token(TokenClass.TERMINAL, "[BOOL]"),
    ">": Token(TokenClass.OPERATOR, ">", operator.gt),
    ">=": Token(TokenClass.OPERATOR, ">=", operator.ge),
    "<": Token(TokenClass.OPERATOR, "<", operator.lt),
    "<=": Token(TokenClass.OPERATOR, "<=", operator.le),
    "==": Token(TokenClass.OPERATOR, "==", operator.eq),
    "!=": Token(TokenClass.OPERATOR, "==", operator.ne),
    "&&": Token(TokenClass.OPERATOR, "&&", operator.and_),
    "||": Token(TokenClass.OPERATOR, "||", operator.or_),
    "IN": Token(TokenClass.OPERATOR, "IN", evaluate_attribute_token),  # operator.contains),
    "(": Token(TokenClass.PARENTHESIS, "("),
    ")": Token(TokenClass.PARENTHESIS, ")"),
}


class TreeNode:
    tokenType = None
    value = None
    left = None
    right = None

    def __init__(self, tokenType: Token):
        self.tokenType: Token = tokenType

    def __str__(self):
        return str(self.value)


class Tokenizer:
    expression: str = None
    tokens: list[str] = None
    tokenTypes: list[Optional[Token]] = None
    i = 0

    def __init__(self, exp):
        self.expression = exp

    def next(self):
        self.i += 1
        return self.tokens[self.i - 1]

    def peek(self):
        return self.tokens[self.i]

    def has_next(self):
        return self.i < len(self.tokens)

    def next_token_type(self):
        return self.tokenTypes[self.i]

    def next_token_type_is_operator(self):
        t = self.tokenTypes[self.i]
        return t.type == TokenClass.OPERATOR

    def tokenize(self):
        import re

        reg = re.compile(r'(&&|\|\||!=|==|<=|>=|<|>|\(|\)|\bIN\b)')
        membership_shorthand = re.compile(r'(\w+):\"?([\w()]+)\"?')
        replaced_expression = membership_shorthand.sub(r'"\2" IN \1', self.expression)
        self.tokens = reg.split(replaced_expression)
        self.tokens = [t.strip() for t in self.tokens if t.strip() != ""]

        self.tokenTypes = []
        for t in self.tokens:
            if t in TOKENS:
                self.tokenTypes.append(TOKENS[t])
            else:
                if t in ('True', 'False'):
                    self.tokenTypes.append(TOKENS["BOOL"])
                elif t[0] == t[-1] == '"' or t[0] == t[-1] == "'":
                    self.tokenTypes.append(TOKENS['STR'])
                else:
                    try:
                        number = float(t)
                        self.tokenTypes.append(TOKENS['NUM'])
                    except:
                        if re.search("^[a-zA-Z_]+$", t):
                            self.tokenTypes.append(TOKENS['VAR'])
                        else:
                            self.tokenTypes.append(None)


class BooleanCondition:
    tokenizer = None
    root = None

    def __init__(self, exp):
        self.tokenizer = Tokenizer(exp)
        self.tokenizer.tokenize()
        self.parse()

    def parse(self):
        self.root = self.parse_expression()

    def parse_expression(self):
        and_term1 = self.parse_and_term()
        while self.tokenizer.has_next() and self.tokenizer.next_token_type() == TOKENS['||']:
            self.tokenizer.next()
            and_term_x = self.parse_and_term()
            and_term = TreeNode(TOKENS['||'])
            and_term.left = and_term1
            and_term.right = and_term_x
            and_term1 = and_term
        return and_term1

    def parse_and_term(self):
        condition1 = self.parse_condition()
        while self.tokenizer.has_next() and self.tokenizer.next_token_type() == TOKENS['&&']:
            self.tokenizer.next()
            condition_x = self.parse_condition()
            condition = TreeNode(TOKENS['&&'])
            condition.left = condition1
            condition.right = condition_x
            condition1 = condition
        return condition1

    def parse_condition(self):
        if self.tokenizer.has_next() and self.tokenizer.next_token_type() == TOKENS['(']:
            self.tokenizer.next()
            expression = self.parse_expression()

            if self.tokenizer.has_next() and self.tokenizer.next_token_type() == TOKENS[')']:
                self.tokenizer.next()
                return expression
            else:
                raise Exception("Closing ) expected, but got " + self.tokenizer.next())

        terminal1 = self.parse_terminal()

        if self.tokenizer.has_next():
            if self.tokenizer.next_token_type_is_operator():
                condition = TreeNode(self.tokenizer.next_token_type())
                self.tokenizer.next()
                terminal2 = self.parse_terminal()
                condition.left = terminal1
                condition.right = terminal2
                return condition
            else:
                raise Exception("Operator expected, but got " + self.tokenizer.next())
        else:
            raise Exception("Operator expected, but got nothing")

    def parse_terminal(self):
        if self.tokenizer.has_next():
            token_type = self.tokenizer.next_token_type()
            if token_type == TOKENS['NUM']:
                n = TreeNode(token_type)
                n.value = float(self.tokenizer.next())
                return n
            elif token_type == TOKENS['VAR'] or token_type == TOKENS['IN']:
                n = TreeNode(token_type)
                n.value = self.tokenizer.next()
                return n
            elif token_type == TOKENS['STR']:
                n = TreeNode(token_type)
                n.value = self.tokenizer.next()[1:-1]
                return n
            elif token_type == TOKENS["BOOL"]:
                n = TreeNode(token_type)
                n.value = self.tokenizer.next() == 'True'
                return n
            else:
                raise Exception("NUM, STR, VAR, IN, BOOL expected, but got " + self.tokenizer.next())

        else:
            raise Exception("NUM, STR, VAR, IN, BOOL expected, but got " + self.tokenizer.next())

    def evaluate(self, variable_dict):
        return self.evaluate_recursive(self.root, variable_dict)

    def evaluate_recursive(self, treeNode, variable_dict):
        if treeNode.tokenType == TOKENS['NUM'] or treeNode.tokenType == TOKENS['STR'] or treeNode.tokenType == TOKENS['BOOL']:
            return treeNode.value
        if treeNode.tokenType == TOKENS['VAR']:
            return variable_dict.get(treeNode.value)

        left = self.evaluate_recursive(treeNode.left, variable_dict)
        right = self.evaluate_recursive(treeNode.right, variable_dict)
        if treeNode.tokenType.type == TokenClass.OPERATOR:
            return TOKENS[treeNode.tokenType.representation].func(left, right)
        else:
            raise Exception("Unexpected type " + str(treeNode.tokenType))

    def to_dnf(self):
        return self.inner_to_dnf(self.root)

    def inner_to_dnf(self, treeNode):
        if treeNode.tokenType == TOKENS['NUM'] or treeNode.tokenType == TOKENS['STR'] or treeNode.tokenType == TOKENS['VAR'] or treeNode.tokenType == TOKENS['BOOL']:
            return treeNode.value

        left = self.inner_to_dnf(treeNode.left)
        right = self.inner_to_dnf(treeNode.right)

        if treeNode.tokenType == TOKENS['&&']:
            return [left, right]

        if treeNode.tokenType == TOKENS['||']:
            return [left], [right]

        if treeNode.tokenType in [TOKENS['>'], TOKENS['>='], TOKENS['<'], TOKENS['<='], TOKENS['=='], TOKENS['!='], TOKENS['IN']]:
            return left, treeNode.tokenType, right
