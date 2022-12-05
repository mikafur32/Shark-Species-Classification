import os

def load_dataset_folder(data_path):
  x = []
  y = []
  for file1 in os.listdir(data_path):
    file2 = os.path.join(data_path, file1)
    for file3 in os.listdir(file2):
        if(file3 == "unused teeth"):
          continue
        else:
          simage = os.path.join(file2,file3)
          x.append(simage)
          y.append(file1)
  return list(x), list(y)