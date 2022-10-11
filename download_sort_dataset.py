import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import csv
import shutil
import numpy as np
import cv2

#download dataset
dataset_url= "https://datasets.ircam.fr/coversdataset/shs_5/f0_cqts_padded_1937x36.tar.gz"
csv_url="https://datasets.ircam.fr/coversdataset/shs_5/examples.csv"
download_url(dataset_url, '.')
download_url(csv_url,".")
print("complete download url")

# Extract from archive
extract_data="./extract_data"
os.mkdir(extract_data)
with tarfile.open('./f0_cqts_padded_1937x36.tar.gz', 'r:gz') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path=extract_data)
print("complete extract to extract_data")
os.remove('./f0_cqts_padded_1937x36.tar.gz')

#sort data each class based on csv 
with open('examples.csv', 'r') as file:
    reader = csv.reader(file)
    list_label = list(reader)


#for make folder each title of cover song
parent= "./data_repo"
os.mkdir(parent)
for a in range(len(list_label)): 
  directory=str(list_label[a][0])
  path = os.path.join(parent, directory)
  os.mkdir(path)

##for move each song(cover) to each title of song
source_parent="./extract_data/f0_cqts_padded_1937x36/"
source_directory_ext= ".f0_cqt.npy"
destination_parent= parent
for i in range(len(list_label)):
  destination_directory= str(list_label[i][0])
  destination= os.path.join(destination_parent,destination_directory)
  for j in range(len(list_label[i])-1): 
    source_directory= str(list_label[i][j+1]) 
    source_directory = source_directory + source_directory_ext
    source = os.path.join(source_parent,source_directory)
    #for resize each file from 1937 to 1024
    reader = np.load(source)
    reader =cv2.resize(reader, dsize=(36, 1024))
    np.save(source, reader)
    #continue move
    shutil.move(source,destination)
    

#COMPLETE
print("complete download and sort")

