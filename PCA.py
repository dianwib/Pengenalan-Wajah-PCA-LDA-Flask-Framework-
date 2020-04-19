import ORL_face
from BAB_II_PENGUKURAN_KEMIRIPAN import manhattan,chebysev,euclidean,minkowski
import numpy as np
from PIL import Image
data_train=ORL_face.data_train
data_test=ORL_face.data_test
# print(data_train.shape,data_test.shape)

class PCA(object):
    def __init__(self,matrix):
        self.matrix=matrix
        zero_mean = self.get_mean()
        covariance = self.get_covariance(zero_mean)
        eigen_value, eigen_vector = self.get_eigen(covariance)
        temp = np.zeros((covariance.shape[0], covariance.shape[0]))

        # eigen value ke 160,160
        for i in range(len(eigen_value)):
            temp[i][i] = eigen_value[i]
        eigen_value = temp

        descending_eigen_vector = self.descending(eigen_value, eigen_vector)
        self.matrix_proyeksi = self.get_proyeksi(descending_eigen_vector, zero_mean)
        self.bobot_train = self.get_bobot(data_train, self.matrix_proyeksi)

        # print("zero main", zero_mean.shape)
        # print("covarience", covariance.shape)
        # print("eigen value", eigen_value.shape)
        # print("eigen vector", eigen_vector.shape)
        # print("descending", descending_eigen_vector.shape)
        # print("mat_proyeksi",self.matrix_proyeksi.shape)
        # print("mat_bobot", self.matrix_bobot.shape)

    def get_mean(self):
        rata_per_kolom=np.mean(self.matrix,axis=0)
        U=np.array([rata_per_kolom,]*self.matrix.shape[0])
        matrix_zero_mean=self.matrix-U
        return matrix_zero_mean


    def get_covariance(self,matrix):
        return (1/matrix.shape[0]-1) * np.dot(matrix,np.transpose(matrix))

    def get_eigen(self,matrix):
        return np.linalg.eig(matrix)

    def descending(self,eigen_value,eigen_vector):
        temp={}
        for i in range(eigen_value.shape[0]):
            temp[i]=eigen_value[i][i]
        temp=sorted(temp.items(), key=lambda k: (k[1]),reverse=True)
        temp_eigen_vector_baru=[]

        for i in range(eigen_vector.shape[0]):
            temp_eigen_vector_baru.append(eigen_vector[temp[i][0]])

        return np.array(temp_eigen_vector_baru)

    def get_proyeksi(self,desc_eigen_vector,zero_mean):
        transpose_zeromain=np.transpose(zero_mean)
        proyeksi=np.dot(transpose_zeromain,desc_eigen_vector)
        return proyeksi

    def get_bobot(self,data_train,matrix_proyeksi):
        bobot=np.dot(data_train,matrix_proyeksi)
        return bobot

    def calc_pca(self,matrix_data_test,metode='man'):

        baris_dimensi_Test=matrix_data_test
        bobot_test=self.get_bobot(baris_dimensi_Test,self.matrix_proyeksi)
        # print(bobot_test.shape,self.bobot_train.shape)


        baris=ORL_face.data.shape[0]#total kelas data dan bukan pose
        kolom=len(ORL_face.list_data_train)
        # print(baris,kolom)
        hasil_matrix=np.zeros((baris,kolom),dtype=float)

        i=0
        for bar in range(baris):
            for kol in range(kolom):
                if metode == 'man':
                    hasil_matrix[bar][kol]=manhattan(self.bobot_train[i],bobot_test)
                elif metode == 'euc':
                    hasil_matrix[bar][kol] = euclidean(self.bobot_train[i], bobot_test)
                elif metode == 'che':
                    hasil_matrix[bar][kol] = chebysev(self.bobot_train[i], bobot_test)
                elif metode == 'min':
                    hasil_matrix[bar][kol] = minkowski (self.bobot_train[i], bobot_test,pangkat=4)

                i+=1
        # print(hasil_manhattan)
        # print("\nMANHATTAN kelas ke :",np.where(hasil_manhattan==np.amin(hasil_manhattan))[0]+1,"objek ke :",np.where(hasil_manhattan==np.amin(hasil_manhattan))[1]+1)
        orang=np.where(hasil_matrix==np.amin(hasil_matrix))[0]+1
        pose=np.where(hasil_matrix==np.amin(hasil_matrix))[1]+1
        return orang,pose


    def eval(self,index_test,orang_hasil):
        orang_test=index_test//len(ORL_face.list_data_test)
        if orang_test== orang_hasil:
            return "benar"
        else:
            return "salah"


# data=ORL_face.data
#index_test=88
# data_test_ke=data_test[index_test]
# orang,pose=PCA(data_train).calc_pca(data_test_ke)
# temp_datatest=np.reshape(data_test_ke,(data.shape[2],data.shape[3]))
# result = Image.new("RGB", (92*2, 112))
#
# for i in range(len(orang)):
#     img_train = (Image.fromarray(data[orang[i]-1][pose[i]-1]))
#     x = i  * 92
#     y = i * 0
#     w, h = img_train.size
#     print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
#     result.paste(img_train, (x, y, x + w, y + h))
#
#     img_test = (Image.fromarray(temp_datatest))
#     x = (i+1) *  92
#     y = (i+1) *  0
#     w, h = img_test.size
#     print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
#     result.paste(img_test, (x, y, x + w, y + h))
#
#
# ##data satu dimensi
# result.show()
