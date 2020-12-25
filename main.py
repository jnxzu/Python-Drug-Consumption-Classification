import pandas as pd

import preprocessing
import association
import classification


def main():
    original_df = preprocessing.read_dataset()
    readable_df = preprocessing.map_to_actual_values(original_df)
    readable_df.to_csv('readable.csv')
    assoc_df = preprocessing.prepare_for_association(original_df)
    assoc_df.to_csv('association.csv')
    class_df = preprocessing.prepare_for_classification(original_df)
    class_df.to_csv('classification.csv')

    assoc_rules = association.find_association_rules(assoc_df)
    interesting_rules = association.get_interesting_rules(assoc_rules)
    interesting_rules.to_csv('interesting_rules.csv')


if __name__ == "__main__":
    main()
