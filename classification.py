from statistics import mean
from time import perf_counter
from numpy.core.fromnumeric import transpose
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense


def naive_bayes_classification(train_in, test_in, train_out, test_out):
    nbc = GaussianNB()
    start_time = perf_counter()
    t_train_out = transpose(train_out)
    t_test_out = transpose(test_out)
    scores = []
    for train_substance, test_substance in zip(t_train_out, t_test_out):
        nbc.fit(train_in, train_substance)
        score = nbc.score(test_in, test_substance)
        scores.append(score)
    avg_score = mean(scores)
    end_time = perf_counter()
    time = end_time - start_time
    return avg_score, time


def knn_three_classification(train_in, test_in, train_out, test_out):
    knn3 = KNeighborsClassifier(n_neighbors=3)
    start_time = perf_counter()
    t_train_out = transpose(train_out)
    t_test_out = transpose(test_out)
    scores = []
    for train_substance, test_substance in zip(t_train_out, t_test_out):
        knn3.fit(train_in, train_substance)
        score = knn3.score(test_in, test_substance)
        scores.append(score)
    avg_score = mean(scores)
    end_time = perf_counter()
    time = end_time - start_time
    return avg_score, time


def knn_five_classification(train_in, test_in, train_out, test_out):
    knn5 = KNeighborsClassifier(n_neighbors=5)
    start_time = perf_counter()
    t_train_out = transpose(train_out)
    t_test_out = transpose(test_out)
    scores = []
    for train_substance, test_substance in zip(t_train_out, t_test_out):
        knn5.fit(train_in, train_substance)
        score = knn5.score(test_in, test_substance)
        scores.append(score)
    avg_score = mean(scores)
    end_time = perf_counter()
    time = end_time - start_time
    return avg_score, time


def knn_eleven_classification(train_in, test_in, train_out, test_out):
    knn11 = KNeighborsClassifier(n_neighbors=11)
    start_time = perf_counter()
    t_train_out = transpose(train_out)
    t_test_out = transpose(test_out)
    scores = []
    for train_substance, test_substance in zip(t_train_out, t_test_out):
        knn11.fit(train_in, train_substance)
        score = knn11.score(test_in, test_substance)
        scores.append(score)
    avg_score = mean(scores)
    end_time = perf_counter()
    time = end_time - start_time
    return avg_score, time


def decision_tree_classification(train_in, test_in, train_out, test_out):
    dtc = DecisionTreeClassifier()
    start_time = perf_counter()
    t_train_out = transpose(train_out)
    t_test_out = transpose(test_out)
    scores = []
    for train_substance, test_substance in zip(t_train_out, t_test_out):
        dtc.fit(train_in, train_substance)
        score = dtc.score(test_in, test_substance)
        scores.append(score)
    avg_score = mean(scores)
    end_time = perf_counter()
    time = end_time - start_time
    return avg_score, time


def custom_neural_network_classification(train_in, test_in, train_out, test_out):
    model = Sequential()
    model.add(Dense(12, input_dim=12, activation='selu'))
    model.add(Dense(13, activation='selu'))
    model.add(Dense(14, activation='selu'))
    model.add(Dense(14, activation='selu'))
    model.add(Dense(15, activation='selu'))
    model.add(Dense(16, activation='selu'))
    model.add(Dense(16, activation='selu'))
    model.add(Dense(17, activation='selu'))
    model.add(Dense(18, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    start_time = perf_counter()
    model.fit(train_in, train_out, validation_data=(
        test_in, test_out), epochs=250, batch_size=10, verbose=0)
    nn_score = model.evaluate(test_in, test_out, verbose=0)[1]
    end_time = perf_counter()
    time = end_time - start_time
    return nn_score, time


def mlp_neural_network_classification(train_in, test_in, train_out, test_out):
    mlp = MLPClassifier(max_iter=250)
    start_time = perf_counter()
    t_train_out = transpose(train_out)
    t_test_out = transpose(test_out)
    scores = []
    for train_substance, test_substance in zip(t_train_out, t_test_out):
        mlp.fit(train_in, train_substance)
        score = mlp.score(test_in, test_substance)
        scores.append(score)
    avg_score = mean(scores)
    end_time = perf_counter()
    time = end_time - start_time
    return avg_score, time


def support_vector_classification(train_in, test_in, train_out, test_out):
    svc = SVC()
    start_time = perf_counter()
    t_train_out = transpose(train_out)
    t_test_out = transpose(test_out)
    scores = []
    for train_substance, test_substance in zip(t_train_out, t_test_out):
        svc.fit(train_in, train_substance)
        score = svc.score(test_in, test_substance)
        scores.append(score)
    avg_score = mean(scores)
    end_time = perf_counter()
    time = end_time - start_time
    return avg_score, time


def random_forest_classification(train_in, test_in, train_out, test_out):
    rfc = RandomForestClassifier()
    start_time = perf_counter()
    t_train_out = transpose(train_out)
    t_test_out = transpose(test_out)
    scores = []
    for train_substance, test_substance in zip(t_train_out, t_test_out):
        rfc.fit(train_in, train_substance)
        score = rfc.score(test_in, test_substance)
        scores.append(score)
    avg_score = mean(scores)
    end_time = perf_counter()
    time = end_time - start_time
    return avg_score, time
