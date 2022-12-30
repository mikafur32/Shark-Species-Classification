import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from Build_Histogram import *

# This file is depreciated as it has been reimplemented in another file
# This file remains to show the progression that occurred from using a linear based SVM to using a non-linear model


def Processing_Pipeline(training_features, testing_features, training_labels, testing_labels):
    descriptors_training = training_features[0]
    for descriptor in training_features[1:]:
        descriptors_training = np.vstack((descriptors_training, descriptor))

    kmeans = KMeans(n_clusters=9, random_state=0)
    kmeans.fit(descriptors_training)

    histograms_training = []
    for descriptor in training_features:
        histogram = build_histogram(descriptor, kmeans)
        histograms_training.append(histogram)

    histograms_testing = []
    for descriptor in testing_features:
        histogram = build_histogram(descriptor, kmeans)
        histograms_testing.append(histogram)

    svm_clf = LinearSVC(max_iter=80000)
    svm_clf.fit(histograms_training, training_labels)

    true_classes = testing_labels
    predicted_classes = []

    for i in svm_clf.predict(histograms_testing):
        predicted_classes.append(i)

    accuracy = accuracy_score(true_classes, predicted_classes)
    return accuracy
