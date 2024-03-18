#Code taken from IsoNet and restructured
from tensorflow.keras.utils import Sequence
import numpy as np
import mrcfile
import os
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class dataSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.x))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        # print('*******',self.x[-1],mrcfile.open(self.x[0]).data[:,:,:,np.newaxis].shape)
        rx = np.array([mrcfile.open(self.x[j]).data[:,:,:,np.newaxis] for j in idx])
        ry = np.array([mrcfile.open(self.y[j]).data[:,:,:,np.newaxis] for j in idx])
        # for j in idx:
        #     print(mrcfile.open(self.x[j]).data.shape,mrcfile.open(self.y[j]).data.shape)
        return rx,ry


# def prepare_dataseq(data_folder, batch_size):

#     dirs_tomake = ['train_x','train_y', 'test_x', 'test_y'] ###['train_xe','train_yo','train_xo','train_ye','test_xe','test_yo','test_xo','test_ye']
#     path_all = []
#     for d in dirs_tomake:
#         p = '{}/{}/'.format(data_folder, d)
#         path_all.append(sorted([p+f for f in os.listdir(p)]))

#     train_data = get_gen(path_all[0], path_all[1], batch_size)
#     test_data = get_gen(path_all[2], path_all[3], batch_size)
#     #train_data_eo = get_gen(path_all[0], path_all[1], batch_size)
#     #train_data_oe = get_gen(path_all[2], path_all[3], batch_size)
#     #test_data_eo = get_gen(path_all[4], path_all[5], batch_size)
#     #test_data_oe = get_gen(path_all[6], path_all[7], batch_size)
#     return train_data, test_data
#     #return train_data_eo, train_data_oe, test_data_eo, test_data_oe

def prepare_dataseq(data_dir, data_folder_odd, data_folder_even, batch_size):

    dirs_tomake_odd = ['train_x', 'test_x'] 
    path_all_odd = []
    for d in dirs_tomake_odd:
        p = '{}/{}/'.format(data_dir, d)
        path_all_odd.append(sorted([p+f for f in os.listdir(p)]))
        
    dirs_tomake_even = ['train_y', 'test_y'] 
    path_all_even = []    
    for d in dirs_tomake_even:
        p = '{}/{}/'.format(data_dir, d)
        path_all_even.append(sorted([p+f for f in os.listdir(p)]))
    
    # if np.random.rand() > 0.5:
    #     train_data = get_gen(path_all_odd[0], path_all_even[0], batch_size)
    #     test_data = get_gen(path_all_odd[1], path_all_even[1], batch_size)
    # else:
    #     train_data = get_gen(path_all_even[0], path_all_odd[0], batch_size)
    #     test_data = get_gen(path_all_even[1], path_all_odd[1], batch_size)

    train_data = get_gen(path_all_odd[0], path_all_even[0], batch_size)
    test_data = get_gen(path_all_odd[1], path_all_even[1], batch_size)

    return train_data, test_data
    
    
def get_gen(x_set,y_set,batch_size,shuffle=True):
    def gen():
        while True:
            all_idx = np.arange(len(x_set))
            if shuffle:
                np.random.shuffle(all_idx)
            for i in range(len(x_set)//batch_size):
                idx = slice(i * batch_size,(i+1) * batch_size)
                idx = all_idx[idx]
                rx = np.array([mrcfile.open(x_set[j]).data[:,:,:,np.newaxis] for j in idx])
                ry = np.array([mrcfile.open(y_set[j]).data[:,:,:,np.newaxis] for j in idx])

                yield rx,ry
    return gen

def get_gen_single(x_set,batch_size,shuffle=True):
    def gen():
        while True:
            all_idx = np.arange(len(x_set))
            if shuffle:
                np.random.shuffle(all_idx)
            for i in range(len(x_set)//batch_size):
                idx = slice(i * batch_size,(i+1) * batch_size)
                idx = all_idx[idx]
                rx = np.array([mrcfile.open(x_set[j]).data[:,:,:,np.newaxis] for j in idx])
                yield rx
    return gen