from mlxtend.frequent_patterns import apriori, association_rules


def find_association_rules(df):
    # apriori algorithm
    freq = apriori(df, min_support=0.0025, use_colnames=True, max_len=3)
    rules = association_rules(freq)             # association rules
    rules = rules[['antecedents', 'consequents', 'support', 'confidence']]
    return rules


def get_interesting_rules(rules):

    # keep only interesting rules
    def filter_antecedents(antes):
        for a in antes:
            if 'True' in a:
                return False
            if 'False' in a:
                return False
        return True

    def filter_consequants(conseqs):
        for c in conseqs:
            if 'True' not in c:
                return False
            if 'caffeine' in c:
                return False
            if 'alcohol' in c:
                return False
            if 'chocolate' in c:
                return False
            if 'nicotine' in c:
                return False
            if 'cannabis' in c:
                return False
        return True

    interesting_rules = rules[(rules['consequents'].apply(
        filter_consequants)) & (rules['antecedents'].apply(filter_antecedents))].sort_values('support', ascending=False)

    return interesting_rules
