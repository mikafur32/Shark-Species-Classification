from Detect_Feature_And_KeyPoints import *
import random

random.seed(0)

def Features_Processing(image_files, labels):
    labels_ = list(labels)
    features_ = []
    for image in image_files:
        keypoints, features = Detect_Feature_And_KeyPoints(cv2.imread(image))
        features_.append(features)
    
    
    num_features = 20
    i = 0
    for feature_set in features_:
        if len(feature_set) < 20:
            features_.pop(i)
            labels_.pop(i)
            i += 1
        else:
            random.shuffle(feature_set)
            feature_set = feature_set[:num_features]
            i += 1
            
    return features_, labels_