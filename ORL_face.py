import os
import numpy as np
from PIL import Image
path_dataset='orl_face_dataset'



temp_folder=[]
for dir_1 in sorted(os.listdir(path_dataset)):
    temp_folder.append(dir_1[1:])
temp_folder.sort(key=int)

data=[]
for dir  in temp_folder:
    temp_file=[]
    for file in os.listdir(path_dataset+'/s'+dir):
        temp_file.append(file[:file.index('.')])
    temp_file.sort(key=int)
    temp=[[] for y in range(len(temp_file))]
    for file in temp_file:
        img=np.array(Image.open(path_dataset+'/s'+dir+'/'+file+'.pgm'))
        temp[int(file)-1]=img
    data.append(temp)

data=np.array(data)

# #PISAH DATA TRAIN DAN TESTING
# DATA TRAIN ukuran 40 x ukuran data training (2,3,6,7,8,10)

list_data_train=[0,1,2]
list_data_test=[3,4,5,6,7,8,9]
#split data train dan testing
matrix_list_data_test=np.delete(data,[list_data_train],1)
matrix_list_data_train=np.delete(data,[list_data_test],1)



#reshape data menjadi orangxface,pxl
def reshape_data(matrix):
    matrix_data = []
    for orang in range(matrix.shape[0]):
        for pose in range(matrix.shape[1]):
            matrix_data.append(np.reshape(matrix[orang][pose], (1, matrix.shape[2]*matrix.shape[3])))
    return np.array(matrix_data)


data_train=reshape_data(matrix_list_data_train)
data_test=reshape_data(matrix_list_data_test)
data_train=np.reshape(data_train,(data_train.shape[0],data_train.shape[2]))
data_test=np.reshape(data_test,(data_test.shape[0],data_test.shape[2]))