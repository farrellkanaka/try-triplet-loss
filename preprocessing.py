import os
from matplotlib.image import imread
import numpy as np
import tables as pt


class PreProcessing:

    images_train = np.array([])
    images_test = np.array([])
    labels_train = np.array([])
    labels_test = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self,data_src):
        self.data_src = data_src
        # Create container
        h5 = pt.open_file('inidata.h5', 'w')
        filters = pt.Filters(complevel=6, complib='blosc')
        print("Loading Dataset...")
        self.images_train, self.images_test, self.labels_train, self.labels_test = self.preprocessing(0.99,h5,filters)
        print("done preproc")
        self.unique_train_label = np.unique(self.labels_train)    #ambil label unique
        self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in
                                        self.unique_train_label}   #dict label uniq dengan setiap label
        print('Preprocessing Done. Summary:')
        print("Images train :", self.images_train.shape)
        print("Labels train :", self.labels_train.shape)
        print("Images test  :", self.images_test.shape)
        print("Labels test  :", self.labels_test.shape)
        print("Unique label :", self.unique_train_label)
        print("labels train : ",self.labels_train)
        print("labels test : ", self.labels_test)
        print("map indices: ",self.map_train_label_indices)

    def normalize(self,x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def read_dataset(self):
        X = []
        y = []
        for directory in sorted(os.listdir(self.data_src)):
            try:
                
                for pic in os.listdir(os.path.join(self.data_src, directory)):
                    img = np.load(os.path.join(self.data_src, directory, pic))
                    X.append(np.asarray(img))
                    y.append(directory)
            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        

        print('Dataset loaded successfully.')
        return X,y

    def preprocessing(self,train_test_ratio,h5,filters):
        X, y = self.read_dataset()
        labels = list(set(y))       #shufling with set
        Y= np.asarray(y)
        shuffle_indices = np.random.permutation(np.arange(len(y)))      #shuffling 2 dari y juga
        x_shuffled = []
        y_shuffled = []
        for index in shuffle_indices:
            x_shuffled.append(X[index])       #x yang sudah teracak urutannya,
            y_shuffled.append(Y[index])       #dan y yang sinkron dengan x
          
        print("ini type Xshuf",type(x_shuffled))
        print("ini shape Xshuf1 ",len(x_shuffled),len(x_shuffled[0]),len(x_shuffled[0][0]))
        print("ini type yshuf",type(x_shuffled))
        print("ini shape yshuf1 ",len(y_shuffled))
        
        print("done shuffling")

        size_of_dataset = len(x_shuffled)
        n_train = int(np.ceil(size_of_dataset * train_test_ratio))

        A = h5.create_carray('/', 'carray1', atom=pt.Float32Atom(), shape=(n_train,len(x_shuffled[0]),len(x_shuffled[0][0],),1), filters=filters)
        B = h5.create_carray('/', 'carray2', atom=pt.Float32Atom(), shape=(size_of_dataset-n_train,len(x_shuffled[0]),len(x_shuffled[0][0]),1), filters=filters)
        print("done make carray")
        
        i=0
        while i < n_train:
          img=np.asarray(X[i])
          img = np.expand_dims(img, axis=-1)
          A[i] = np.asarray(img)
          i+=1

        i= n_train
        while i < size_of_dataset:
          img= np.asarray(X[i])
          img=np.expand_dims(img,axis=-1)
          B[i-n_train] = np.asarray(img)
          i+=1
        

        return A,B,np.asarray(Y[0:n_train]), np.asarray(Y[n_train + 1:size_of_dataset])


    def get_triplets(self):
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False)
        a, p = np.random.choice(self.map_train_label_indices[label_l],2, replace=False)
        n = np.random.choice(self.map_train_label_indices[label_r])
        return a, p, n

    def get_triplets_batch(self,n):
        idxs_a, idxs_p, idxs_n = [], [], []
        for num in range(n):
          a, p, n = self.get_triplets()

              #check if apn have duplicate
          for nom in range(len(idxs_a)):
            
            if idxs_a[nom]==a or idxs_p[nom]==p or idxs_n[nom]==n:
              num=num-1         #if there duplicate, repeat it with -1 iter
              break 
          else:
              idxs_a.append(a)
              idxs_p.append(p)
              idxs_n.append(n)
            #point break continue
        return self.images_train[idxs_a,:], self.images_train[idxs_p, :], self.images_train[idxs_n, :]

