from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from Build_Histogram import *


def Testing_Poly_SVC(training_features, testing_features, training_labels, testing_labels, c, d, k):
    descriptors_training = training_features[0]
    for descriptor in training_features[1:]:
        descriptors_training = np.vstack((descriptors_training, descriptor))

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(descriptors_training)
    """
    centroids = kmeans.cluster_centers_
    dist_to_centroids = kmeans.transform(descriptors_training)

    dists = [[] for _ in range(k)]

    for im in dist_to_centroids:
      for i in range(k):
        dists[i].append(im[i])

    rep_ims = []
    rep_ims_labels = []
    for dist in dists:
      rep_im = training_features[np.argmin(dist)]
      rep_im_label = training_labels[np.argmin(dist)]
      rep_ims.append(rep_im)
      rep_ims_labels.append(rep_im_label)
    """
    histograms_training = []
    for descriptor in training_features:
        histogram = build_histogram(descriptor, kmeans)
        histograms_training.append(histogram)

    histograms_testing = []
    for descriptor in testing_features:
        histogram = build_histogram(descriptor, kmeans)
        histograms_testing.append(histogram)

    svm_clf = SVC(kernel="poly", degree=d, C=c, coef0=1, random_state=0)
    svm_clf.fit(histograms_training, training_labels)
    testing_predictions = svm_clf.predict(histograms_testing)  # create predictions based of the polynomial classifier

    accuracy = svm_clf.score(testing_predictions, testing_labels)
    #accuracy = accuracy_score(testing_predictions, testing_labels)

    return accuracy
