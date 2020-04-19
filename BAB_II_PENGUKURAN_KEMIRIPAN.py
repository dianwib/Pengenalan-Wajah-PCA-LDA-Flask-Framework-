import numpy as np
import math
# matrix_X_1_1=np.array([[0,9,0],[3,5,8],[7,4,3],[6,3,3],[2,1,5],[10,8,7],[5,1,2],[3,4,2],[6,4,10],[9,9,6]])
# matrix_X_1_2=np.array([[3,6,6],[10,5,3],[9,0,8],[2,7,8],[0,9,8],[7,9,4],[9,1,1],[1,4,7],[0,2,10],[6,6,2]])
#
# matrix_X_2_1=np.array([[1,5,4],[9,0,1],[5,10,9],[9,5,3],[8,3,3],[2,2,0],[2,6,8],[1,6,5],[6,8,5],[8,2,9]])
# matrix_X_2_2=np.array([[7,0,5],[7,5,4],[2,0,9],[6,5,7],[10,10,7],[6,2,9],[8,4,5],[10,4,8],[9,8,5],[2,8,8]])
#
# matrix_X_3_1=np.array([[4,3,9],[7,8,2],[10,5,7],[5,5,9],[4,8,2],[2,1,3],[3,6,5],[1,3,9],[4,3,5],[3,6,3]])
# matrix_X_3_2=np.array([[5,2,2],[6,8,8],[4,4,1],[0,1,10],[7,4,2],[5,8,3],[4,2,7],[1,9,1],[4,3,5],[2,3,7]])
#
# matrix_X_4_1=np.array([[9,8,5],[7,5,9],[7,3,1],[4,7,10],[2,10,1],[5,8,5],[2,8,4],[8,7,1],[7,6,7],[8,8,6]])
# matrix_X_4_2=np.array([[8,3,5],[6,10,8],[8,3,10],[6,1,1],[9,8,7],[9,8,2],[5,9,4],[8,3,3],[5,8,8],[8,5,10]])
#
# matrix_X_5_1=np.array([[6,5,1],[7,5,4],[9,9,1],[7,2,6],[6,3,5],[4,6,8],[6,5,2],[4,3,6],[4,6,5],[4,9,9]])
# matrix_X_5_2=np.array([[7,7,10],[4,0,9],[3,4,1],[3,9,3],[10,3,2],[3,6,3],[2,3,10],[4,1,5],[7,4,7],[7,8,8]])
#
# matrix_Y=np.array([[7,1,10],[3,6,10],[5,2,2],[0,1,8],[9,6,6],[4,3,9],[3,4,6],[10,7,7],[5,9,7],[3,3,4]])
#
# matrix_X=np.array([[matrix_X_1_1,matrix_X_1_2],[matrix_X_2_1,matrix_X_2_2],[matrix_X_3_1,matrix_X_3_2],[matrix_X_4_1,matrix_X_4_2],[matrix_X_5_1,matrix_X_5_2]])

def chebysev(matrix_1,matrix_2):
    temp_hasil=np.absolute(matrix_1-matrix_2)
    temp_hasil=np.amax(temp_hasil)
    return temp_hasil

def manhattan(matrix_1,matrix_2):
    temp_hasil=np.absolute(matrix_1-matrix_2)
    temp_hasil=np.sum(temp_hasil)
    return temp_hasil

def euclidean(matrix_1,matrix_2):
    temp_hasil=np.power((matrix_1-matrix_2),2)
    temp_hasil=np.sum(temp_hasil)
    temp_hasil=np.sqrt(temp_hasil)
    return temp_hasil


def minkowski(matrix_1,matrix_2,pangkat):
    temp_hasil = np.power((matrix_1 - matrix_2), pangkat)
    temp_hasil = np.sum(temp_hasil)
    temp_hasil = temp_hasil**(1/pangkat)
    return temp_hasil

def angular_separation(matrix_1,matrix_2):

    temp_hasil_atas = np.array(matrix_1*matrix_2)
    temp_hasil_atas=np.sum(temp_hasil_atas)

    kiri=np.sum(np.power(matrix_1,2))
    kanan = np.sum(np.power(matrix_2, 2))

    temp_hasil_bawah=np.array(kanan*kiri)
    temp_hasil_bawah=np.sqrt(temp_hasil_bawah)

    temp_hasil = temp_hasil_atas/temp_hasil_bawah
    return temp_hasil



#
# #print(matrix_X.shape)
# bar,kol=matrix_X.shape[:2]
# print(bar,kol)
# hasil_chebysev=np.zeros((bar,kol),dtype=float)
# hasil_manhattan=np.zeros((bar,kol),dtype=float)
# hasil_euclidean=np.zeros((bar,kol),dtype=float)
# hasil_minkowski=np.zeros((bar,kol),dtype=float)
# hasil_angularseparation=np.zeros((bar,kol),dtype=float)
# for baris in range(bar):
#     for kolom in range(kol):
#         hasil_chebysev[baris][kolom] = chebysev(matrix_X[baris][kolom], matrix_Y)
#         hasil_manhattan[baris][kolom]=manhattan(matrix_X[baris][kolom],matrix_Y)
#         hasil_euclidean[baris][kolom] = euclidean(matrix_X[baris][kolom], matrix_Y)
#         hasil_minkowski[baris][kolom] = minkowski(matrix_X[baris][kolom], matrix_Y,4)
#         hasil_angularseparation[baris][kolom] = angular_separation(matrix_X[baris][kolom], matrix_Y)
#
# #
# # print(np.where(hasil_manhattan==np.amin(hasil_manhattan))[0])
# print(hasil_chebysev,"\nCHEBYSEV kelas ke :",np.where(hasil_chebysev==np.amin(hasil_chebysev))[0]+1,"objek ke :",np.where(hasil_chebysev==np.amin(hasil_chebysev))[1]+1)
# print(hasil_manhattan,"\nMANHATTAN kelas ke :",np.where(hasil_manhattan==np.amin(hasil_manhattan))[0]+1,"objek ke :",np.where(hasil_manhattan==np.amin(hasil_manhattan))[1]+1)
# print(hasil_euclidean,"\nEUCLIDEAN kelas ke :",np.where(hasil_euclidean==np.amin(hasil_euclidean))[0]+1,"objek ke :",np.where(hasil_euclidean==np.amin(hasil_euclidean))[1]+1)
# print(hasil_minkowski,"\nMINKOWSKI kelas ke :",np.where(hasil_minkowski==np.amin(hasil_minkowski))[0]+1,"objek ke :",np.where(hasil_minkowski==np.amin(hasil_minkowski))[1]+1)
# #khusus angular amax
# print(hasil_angularseparation,"\nANGULAR SEPARATION kelas ke :",np.where(hasil_angularseparation==np.amax(hasil_angularseparation))[0]+1,"objek ke :",np.where(hasil_angularseparation==np.amax(hasil_angularseparation))[1]+1)
