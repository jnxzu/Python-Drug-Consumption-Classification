import classification
import association
import preprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    original_df = preprocessing.read_dataset()
    readable_df = preprocessing.map_to_actual_values(original_df)
    assoc_df = preprocessing.prepare_for_association(original_df)
    train_in, test_in, train_out, test_out = preprocessing.prepare_for_classification(
        original_df)

    readable_df.to_csv('readable.csv')

    assoc_rules = association.find_association_rules(assoc_df)
    interesting_rules = association.get_interesting_rules(assoc_rules)
    interesting_rules.to_csv('interesting_rules.csv')

    nbc_score, nbc_time = classification.naive_bayes_classification(
        train_in, test_in, train_out, test_out)

    knn3_score, knn3_time = classification.knn_three_classification(
        train_in, test_in, train_out, test_out)

    knn5_score, knn5_time = classification.knn_five_classification(
        train_in, test_in, train_out, test_out)

    knn11_score, knn11_time = classification.knn_eleven_classification(
        train_in, test_in, train_out, test_out)

    dtc_score, dtc_time = classification.decision_tree_classification(
        train_in, test_in, train_out, test_out)

    cnn_score, cnn_time = classification.custom_neural_network_classification(
        train_in, test_in, train_out, test_out)

    mlp_score, mlp_time = classification.mlp_neural_network_classification(
        train_in, test_in, train_out, test_out)

    svm_score, svm_time = classification.support_vector_classification(
        train_in, test_in, train_out, test_out)

    rfc_score, rfc_time = classification.random_forest_classification(
        train_in, test_in, train_out, test_out)


if __name__ == "__main__":
    main()
