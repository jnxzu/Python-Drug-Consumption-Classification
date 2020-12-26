import classification
import association
import preprocessing
import os

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    original_df = preprocessing.read_dataset()
    readable_df = preprocessing.map_to_actual_values(original_df)
    assoc_df = preprocessing.prepare_for_association(original_df)
    train_in, test_in, train_out, test_out = preprocessing.prepare_for_classification(
        original_df)

    readable_df.to_csv('readable.csv')

    assoc_rules = association.find_association_rules(assoc_df)
    substance_rules, other_rules = association.get_interesting_rules(
        assoc_rules)
    substance_rules.to_csv('substance_rules.csv')
    other_rules.to_csv('other_rules.csv')

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

    labels = ['NB', 'kNN3', 'kNN5',
              'kNN11', 'DTC', 'NN', 'MLP', 'SVM', 'RFC']
    scores = [nbc_score, knn3_score, knn5_score, knn11_score,
              dtc_score, cnn_score, mlp_score, svm_score, rfc_score]
    scores = list(map(lambda x: round(x*100, 1), scores))
    times = [nbc_time, knn3_time, knn5_time, knn11_time,
             dtc_time, cnn_time, mlp_time, svm_time, rfc_time]

    plot_space = max(scores) - min(scores)
    plt.ylim([min(scores) - plot_space, max(scores) + plot_space])
    plt.bar(labels, scores)
    for i, v in enumerate(scores):
        plt.text(i - .3, v + 3, str(v))
    plt.xlabel('Classifier')
    plt.ylabel('% Accuracy')
    plt.title('Classifier Accuracy Comparison - Drug Consumption')
    plt.savefig('classifier_accuracy.png')
    plt.close()

    plot_space = max(times) - min(times)
    plt.bar(labels, times)
    plt.xlabel('Classifier')
    plt.ylabel('Time (s)')
    plt.title('Classifier Time Comparison - Drug Consumption')
    plt.savefig('classifier_timing.png')
    plt.close()


if __name__ == "__main__":
    main()
