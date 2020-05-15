from PCA import PCA
import ORL_face
import  numpy as np
from PIL import Image
from BAB_II_PENGUKURAN_KEMIRIPAN import manhattan,chebysev,euclidean,minkowski
data_train=ORL_face.data_train #240x10304
data_test=ORL_face.data_test  #160x10304






class LDA(object):
    def __init__(self,index_test):

        data_test_ke = data_test[int(index_test)]
        pca = PCA(data_train)
        orang, pose = pca.calc_pca(data_test_ke)
        Input_LDA = {}
        Input_LDA['bobot'] = pca.bobot_train
        Input_LDA['proyeksi'] = pca.matrix_proyeksi
        Input_LDA['jumlah_kelas'] = ORL_face.data.shape[0]  # jumlah semua kelas(40) bukan pose
        Input_LDA['jumlah_pose'] = len(ORL_face.list_data_train)  # jumlah semua  pose train
        Input_LDA['data_train'] = data_train
        self.input_LDA=Input_LDA

        jumlah_kelas=self.input_LDA['jumlah_kelas']
        jumlah_pose_train=self.input_LDA['jumlah_pose']
        jumlah_data=jumlah_kelas*jumlah_pose_train

        self.proyeksi_pca_baru=self.get_proyeksi_pca_baru(self.input_LDA['proyeksi'],jumlah_data,jumlah_kelas)
        self.input_LDA=self.get_input_LDA(self.input_LDA['data_train'],self.proyeksi_pca_baru)
        self.rata_per_kelas=self.get_rata_tiap_kelas(self.input_LDA,jumlah_kelas,jumlah_pose_train)
        self.rata_total_kelas=self.get_rata_total_kelas(self.input_LDA)
        self.Sb=self.get_between_class_scatter(self.rata_per_kelas,self.rata_total_kelas,jumlah_kelas)
        self.Sw=self.get_within_class_scatter(self.input_LDA,self.rata_per_kelas,jumlah_data,jumlah_kelas,jumlah_pose_train)
        self.eigen_value, self.eigen_vector = self.get_eigen(self.Sb,self.Sw)
        self.descending_eigen_vector = self.descending(self.eigen_value, self.eigen_vector)
        self.wFid=np.transpose(self.descending_eigen_vector[:,0:jumlah_kelas-1])
        self.proyeksi=self.get_proyeksi(self.wFid,self.proyeksi_pca_baru)
        self.bobot_train= self.get_bobot(data_train, self.proyeksi)
        print("\nLDA","=="*30)
        print("proyeksi lama",self.proyeksi_pca_baru.shape)
        print("input LDA",self.input_LDA.shape)
        print("rata per kelas",self.rata_per_kelas.shape)
        print("rata semua kelas",self.rata_total_kelas.shape)
        print("Sb",self.Sb.shape)
        print("Sw",self.Sw.shape)
        print("eva",self.eigen_value.shape)
        print("eve",self.eigen_vector.shape,self.descending_eigen_vector.shape)
        print("wFid",self.wFid.shape)
        print("proyeksi",self.proyeksi.shape)
        print("bobot",self.bobot_train.shape)

    def get_proyeksi_pca_baru(self,input_proyeksi,jumlah_data,jumlah_kelas):
        proyeksi_pca_baru=input_proyeksi[:,0:jumlah_data-jumlah_kelas]
        return proyeksi_pca_baru

    def get_input_LDA(self,data_train,proyeksi_pca_baru):
        data_train=np.array(data_train,dtype=float)
        input_LDA=np.dot(data_train,proyeksi_pca_baru)
        return input_LDA

    def get_rata_tiap_kelas(self,data_input_lda,jumlah_kelas,jumlah_pose):
        rata=np.zeros((jumlah_kelas,data_input_lda.shape[1]))
        for i in range(jumlah_kelas):
            mulai=i
            sampai=i+jumlah_pose-1
            rata[i,:]=np.mean(data_input_lda[mulai:sampai,:])
        return rata

    def get_rata_total_kelas(self,data_input_lda):
        rata_total=np.mean(data_input_lda,axis=0)
        return rata_total

    def get_between_class_scatter(self,rata_per_kelas,rata_total_kelas,jumlah_kelas):
        # sb=np.zeros((jumlah_kelas,rata_total_kelas.shape[0]))
        sb=0
        for i in range(jumlah_kelas):
            zk=rata_per_kelas[i,:]-rata_total_kelas
            sb=sb+np.dot(np.transpose(zk),zk)
        return sb

    def get_within_class_scatter(self,data_input_lda,rata_per_kelas,jumlah_data,jumlah_kelas,jumlah_pose):
        # sw=np.zeros((rata_per_kelas.shape[0]*jumlah_data,rata_per_kelas.shape[1]))
        index=0
        sw=0
        for i in range(jumlah_kelas):
            for j in range(jumlah_data):
                zm=data_input_lda[j,:]-rata_per_kelas[i:]
                # print(zm.shape)
                sw=sw +np.dot(jumlah_pose,  np.dot(np.transpose(zm),zm))
        return sw

    def get_eigen(self,Sb,Sw):
        temp=np.dot(Sb,np.linalg.inv(Sw))
        return np.linalg.eig(temp)

    def descending(self,eigen_value,eigen_vector):

        temp = np.zeros((eigen_value.shape[0], eigen_value.shape[0]))

        # eigen value ke 160,160
        for i in range(len(eigen_value)):
            temp[i][i] = eigen_value[i]
        eigen_value = temp

        temp={}
        for i in range(eigen_value.shape[0]):
            temp[i]=eigen_value[i][i]
        temp=sorted(temp.items(), key=lambda k: (k[1]),reverse=True)
        temp_eigen_vector_baru=[]

        for i in range(eigen_vector.shape[0]):
            temp_eigen_vector_baru.append(eigen_vector[temp[i][0]])

        return np.array(temp_eigen_vector_baru)

    def get_proyeksi(self,wFid,proyeksi_pca_baru):
        transpose_proyeksi_pca_baru=np.transpose(proyeksi_pca_baru)
        proyeksi=np.transpose(np.dot(wFid,transpose_proyeksi_pca_baru))
        return proyeksi

    def get_bobot(self,data_train,matrix_proyeksi):
        bobot=np.dot(np.array(data_train,dtype=float),matrix_proyeksi)
        return bobot

    def calc_lda(self, matrix_data_test, metode='man'):

        baris_dimensi_Test = matrix_data_test
        bobot_test = self.get_bobot(baris_dimensi_Test, self.proyeksi)
        # print(bobot_test.shape,self.bobot_train.shape)

        baris = ORL_face.data.shape[0]  # total kelas data dan bukan pose
        kolom = len(ORL_face.list_data_train)
        # print(baris,kolom)
        hasil_matrix = np.zeros((baris, kolom), dtype=float)

        i = 0
        for bar in range(baris):
            for kol in range(kolom):
                if metode == 'man':
                    hasil_matrix[bar][kol] = manhattan(self.bobot_train[i], bobot_test)
                elif metode == 'euc':
                    hasil_matrix[bar][kol] = euclidean(self.bobot_train[i], bobot_test)
                elif metode == 'che':
                    hasil_matrix[bar][kol] = chebysev(self.bobot_train[i], bobot_test)
                elif metode == 'min':
                    hasil_matrix[bar][kol] = minkowski(self.bobot_train[i], bobot_test, pangkat=4)

                i += 1
        # print(hasil_manhattan)
        # print("\nMANHATTAN kelas ke :",np.where(hasil_manhattan==np.amin(hasil_manhattan))[0]+1,"objek ke :",np.where(hasil_manhattan==np.amin(hasil_manhattan))[1]+1)
        orang = np.where(hasil_matrix == np.amin(hasil_matrix))[0] + 1
        pose = np.where(hasil_matrix == np.amin(hasil_matrix))[1] + 1
        return orang, pose

# print((data_train.dtype))
# a=LDA(Input_LDA)
# print("proyeksi",a.proyeksi_pca_baru.shape)
# print("input LDA",a.input_LDA.shape)
# print("rata per kelas",a.rata_per_kelas.shape)
# print("rata semua kelas",a.rata_total_kelas.shape)
# print("Sb",a.Sb.shape)
# print("Sw",a.Sw.shape)
# print("eva",a.eigen_value.shape)
# print("eve",a.eigen_vector.shape,a.descending_eigen_vector.shape)
# print("wFid",a.wFid.shape)
# print("proyeksi",a.proyeksi.shape)
# print("bobot",a.bobot_train.shape)
#
#
# data=ORL_face.data
# index_test=2
# data_test_ke=data_test[index_test]
# orang,pose=LDA(index_test).calc_lda(data_test_ke)
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

