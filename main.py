from Build_Histogram import *
from Detect_Feature_And_KeyPoints import *
from Load_Dataset_Folder import *
from Features_Processing import *
from Linear_Processsing_Pipeline import *
from Training_Poly_Processing_Pipeline import *
from Testing_Poly_Processing_Pipeline import *
from LeNet_Implementation import *
from sklearn.model_selection import train_test_split
import torch.utils.data as data

user= 'perso'
root_path = "C:\\Users\\" + user +"\\Documents\\GitHub\\Shark-Species-Classification"
data_path = os.path.join(root_path, 'Genus Carcharhinus')


dataset = ImageLoader(data_path)

train_dataset, test_dataset = Dataset_Splitter(.5, dataset)
train_train_dataset, validation_dataset = Dataset_Splitter(.9, train_dataset)

BATCH_SIZE = 64

train_iterator = data.DataLoader(train_train_dataset,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(validation_dataset,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_dataset,
                                batch_size=BATCH_SIZE)


criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
optimizer = optim.Adam(model.parameters())
model = model.to(device)
criterion = criterion.to(device)
"""
"""
image_files, labels = load_dataset_folder(data_path)
features, processed_labels = Features_Processing(image_files, labels)


features_train, features_test, labels_train, labels_test = train_test_split(features, processed_labels, test_size = .2, random_state = 0)
features_train_train, features_validation, labels_train_train, labels_validation = train_test_split(features_train, labels_train, test_size = .25, random_state = 0)


c, d = Poly_Processing_Pipeline(features_train_train, features_validation, labels_train_train, labels_validation)


training_accuracy = Testing_Poly_SVC(features_train_train, features_validation, labels_train_train, labels_validation, c, d)
testing_accuracy = Testing_Poly_SVC(features_train, features_test, labels_train, labels_test,c, d)

print("c: ", c)
print("d: ", d)
print("Training Accuracy: ", training_accuracy)
print("Testing Accuracy: ", testing_accuracy)
"""