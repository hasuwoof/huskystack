from huskygen.filter.boolean_condition import BooleanCondition


if __name__ == "__main__":
    # tk = Tokenizer("species:cat && rating:safe && (meta:width > 1000 || meta:height > 1000)")
    # tk.tokenize()

    e621_test = BooleanCondition("species:dog && rating == 'safe'")
    post = {id: 1, "species": ["cat", "dog"], "rating": "safe", "width": 100, "height": 100}
    print(e621_test.to_dnf())
    print(e621_test.evaluate(post))
    # assert e621_test.evaluate(post) == True
    # Bug fix in matching string + Include both "<double quoted>" and '<single quoted>' string
    # double_quoted_p = BooleanParser('account_number == "abc"')
    # assert double_quoted_p.evaluate({'account_number': 'abc'}) == True

    # single_quoted_p = BooleanParser("account_number == 'abc'")
    # assert single_quoted_p.evaluate({'account_number': 'abc'}) == True
    # assert single_quoted_p.evaluate({'account_number': "abc"}) == True
