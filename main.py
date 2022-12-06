from torch import optim
from tqdm.notebook import tqdm, trange
from Build_Histogram import *
from Detect_Feature_And_KeyPoints import *
from Load_Dataset_Folder import *
from Features_Processing import *
from Linear_Processsing_Pipeline import *
from Training_Poly_Processing_Pipeline import *
from Testing_Poly_Processing_Pipeline import *
from calculate_accuracy import *
from train import *
from evaluate import *
from LeNet_Implementation import *
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import torch

print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)
x = torch.randn(1).cuda()
print(x)


'''
print(torch.cuda.is_available())

print(torch.cuda.get_device_capability())
'''

user= 'perso'

root_path = "C:\\Users\\" + user + "\\Documents\\GitHub\\Shark-Species-Classification"
data_path = os.path.join(root_path, 'Genus Carcharhinus')


dataset = ImageLoader(data_path)

train_dataset, test_dataset = Dataset_Splitter(.8, dataset)
train_train_dataset, validation_dataset = Dataset_Splitter(.8, train_dataset)

train_dataset, test_dataset = Dataset_Splitter(.5, dataset)
train_train_dataset, validation_dataset = Dataset_Splitter(.9, train_dataset)
'''

BATCH_SIZE = 64

train_iterator = data.DataLoader(train_train_dataset,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(validation_dataset,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_dataset,
                                batch_size=BATCH_SIZE)

OUTPUT_DIM = 9
model = MMNet(OUTPUT_DIM)

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda') #if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters())
model = model.to(device)
criterion = criterion.to(device)

EPOCHS = [1,3,5,10,15,20,30]
EPOCHS = 7
best_epoch = 0

best_valid_loss = float('inf')
'''

'''
for epoch_ in trange(EPOCHS, desc="Epochs"):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'MMnet-model.pt')
        best_epoch = epoch_

    print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

'''

'''
#model.load_state_dict(torch.load('MMnet-model.pt'))

#test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

#print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

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
'''