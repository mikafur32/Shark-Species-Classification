import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from Build_Histogram import *


def Poly_SVC(training_features, testing_features, training_labels, testing_labels):
    Cs = [1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2, 1E3]
    ds = [1, 2, 3, 4]
    val_accuracy = 0
    best_c_d = (0, 0)

    for i in Cs:
        for j in ds:
            svm_clf = SVC(kernel="poly", degree=j, C=i, coef0=1, random_state=0)
            svm_clf.fit(training_features, training_labels)
            validation_predictions = svm_clf.predict(
                testing_features)  # create predictions based of the polynomial classifier
            if accuracy_score(validation_predictions, testing_labels) >= val_accuracy:
                val_accuracy = accuracy_score(validation_predictions, testing_labels)
                returns = (i,j,val_accuracy)


    return returns





def Poly_Processing_Pipeline(training_features, testing_features, training_labels, testing_labels):
    descriptors_training = training_features[0]
    for descriptor in training_features[1:]:
        descriptors_training = np.vstack((descriptors_training, descriptor))

    clusters = [9,10,11,12,13,14,15]

    best_val = np.inf
    best_k = 0
    c, d = 0,0
    best_c, best_d = 0,0
    for k in clusters:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(descriptors_training)

        histograms_training = []
        for descriptor in training_features:
            histogram = build_histogram(descriptor, kmeans)
            histograms_training.append(histogram)

        histograms_testing = []
        for descriptor in testing_features:
            histogram = build_histogram(descriptor, kmeans)
            histograms_testing.append(histogram)

        c,d,val = Poly_SVC(histograms_training, histograms_testing, training_labels, testing_labels)

        #returns = Poly_SVC(histograms_training, histograms_testing, training_labels, testing_labels)
        #c, d, val = returns[0], returns[1], returns[2]


        if val < best_val:
            best_val = val
            best_k = k
            best_c = c
            best_d = d

    return best_c, best_d, best_k

