from flask import Flask, render_template, request
from PCA import PCA
from LDA import LDA
import ORL_face
import numpy as np
from PIL import Image



app = Flask(__name__)
@app.route('/')
@app.route('/index')
def hello_world():
	return render_template('index.html')

import datetime

@app.route('/PCA',methods=['GET', 'POST'])
def pca():
    path_hasil = 'static/assets/PCA/hasil/'

    if request.method == 'POST':
        id_time=datetime.datetime.now().time().microsecond

        input_data_test_ke = (request.form['datatest_input'])
        pilih_metode = str(request.form['pilih_metode'])

        # print(pilih_metode,type(pilih_metode))
        data = ORL_face.data
        data_train = ORL_face.data_train
        data_test = ORL_face.data_test

        if input_data_test_ke != "SEMUA":
            data_test_ke = data_test[int(input_data_test_ke)]
            orang, pose = PCA(data_train).calc_pca(data_test_ke,pilih_metode)
            temp_datatest = np.reshape(data_test_ke, (data.shape[2], data.shape[3]))
            result = Image.new("RGB", (92 * 2, 112))

            img_train = (Image.fromarray(data[orang[0] - 1][pose[0] - 1]))
            x = 0
            y = 0
            w, h = img_train.size
            # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
            result.paste(img_train, (x, y, x + w, y + h))

            img_test = (Image.fromarray(temp_datatest))
            x = (1) * 92
            y = (1) * 0
            w, h = img_test.size
            # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
            result.paste(img_test, (x, y, x + w, y + h))
            filename_img = str(id_time)+"_"+pilih_metode+"_" + str(1)
            result.save('static/assets/PCA/hasil/' + filename_img + '.png')
            hasil_keterangan=PCA.eval(None,int(input_data_test_ke),orang-1)
            return render_template('PCA.html',query_path=input_data_test_ke,img_hasil=filename_img,keterangan=hasil_keterangan)

        elif input_data_test_ke == "SEMUA":
            temp_benar=0
            temp_salah=0
            temp_keterangan=[]
            temp_img=[]
            for input_data_test_ke in range ((ORL_face.data_test.shape[0])):
                data_test_ke = data_test[int(input_data_test_ke)]
                orang, pose = PCA(data_train).calc_pca(data_test_ke,pilih_metode)
                temp_datatest = np.reshape(data_test_ke, (data.shape[2], data.shape[3]))
                result = Image.new("RGB", (92 * 2, 112))

                img_train = (Image.fromarray(data[orang[0] - 1][pose[0] - 1]))
                x = 0
                y = 0
                w, h = img_train.size
                # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
                result.paste(img_train, (x, y, x + w, y + h))

                img_test = (Image.fromarray(temp_datatest))
                x = (1) * 92
                y = (1) * 0
                w, h = img_test.size
                # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
                result.paste(img_test, (x, y, x + w, y + h))
                filename_img = str(id_time) + "__"+pilih_metode+"_" + str(input_data_test_ke+1)
                result.save('static/assets/PCA/hasil/' + filename_img + '.png')
                hasil_keterangan = PCA.eval(None, int(input_data_test_ke), orang - 1)
                temp_keterangan.append(hasil_keterangan)
                temp_img.append(filename_img)
                if hasil_keterangan == 'benar':
                    temp_benar+=1
                else:
                    temp_salah+=1

            hasil_akurasi=(temp_benar/(temp_benar+temp_salah))*100
            return render_template('PCA.html',query_path="SEMUA",len_data_test=ORL_face.data_test.shape[0],img=temp_img,akurasi=hasil_akurasi,keterangan=temp_keterangan)
    else:
        return render_template('PCA.html')



#LDA

@app.route('/LDA',methods=['GET', 'POST'])
def lda():

    path_hasil = 'static/assets/LDA/hasil/'

    if request.method == 'POST':
        id_time=datetime.datetime.now().time().microsecond

        input_data_test_ke = (request.form['datatest_input'])
        pilih_metode = str(request.form['pilih_metode'])

        # print(pilih_metode,type(pilih_metode))
        data = ORL_face.data
        data_train = ORL_face.data_train
        data_test = ORL_face.data_test

        if input_data_test_ke != "SEMUA":

            data_test_ke = data_test[int(input_data_test_ke)]
            orang, pose = LDA(input_data_test_ke).calc_lda(data_test_ke,pilih_metode)
            temp_datatest = np.reshape(data_test_ke, (data.shape[2], data.shape[3]))
            result = Image.new("RGB", (92 * 2, 112))

            img_train = (Image.fromarray(data[orang[0] - 1][pose[0] - 1]))
            x = 0
            y = 0
            w, h = img_train.size
            # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
            result.paste(img_train, (x, y, x + w, y + h))

            img_test = (Image.fromarray(temp_datatest))
            x = (1) * 92
            y = (1) * 0
            w, h = img_test.size
            # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
            result.paste(img_test, (x, y, x + w, y + h))
            filename_img = str(id_time)+"_"+pilih_metode+"_" + str(1)
            result.save('static/assets/LDA/hasil/' + filename_img + '.png')
            hasil_keterangan=PCA.eval(None,int(input_data_test_ke),orang-1)
            return render_template('LDA.html',query_path=input_data_test_ke,img_hasil=filename_img,keterangan=hasil_keterangan)

        elif input_data_test_ke == "SEMUA":
            temp_benar=0
            temp_salah=0
            temp_keterangan=[]
            temp_img=[]
            for input_data_test_ke in range ((ORL_face.data_test.shape[0])):
                data_test_ke = data_test[int(input_data_test_ke)]
                orang, pose = LDA(input_data_test_ke).calc_lda(data_test_ke, pilih_metode)
                temp_datatest = np.reshape(data_test_ke, (data.shape[2], data.shape[3]))
                result = Image.new("RGB", (92 * 2, 112))

                img_train = (Image.fromarray(data[orang[0] - 1][pose[0] - 1]))
                x = 0
                y = 0
                w, h = img_train.size
                # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
                result.paste(img_train, (x, y, x + w, y + h))

                img_test = (Image.fromarray(temp_datatest))
                x = (1) * 92
                y = (1) * 0
                w, h = img_test.size
                # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
                result.paste(img_test, (x, y, x + w, y + h))
                filename_img = str(id_time) + "__"+pilih_metode+"_" + str(input_data_test_ke+1)
                result.save('static/assets/LDA/hasil/' + filename_img + '.png')
                hasil_keterangan = PCA.eval(None, int(input_data_test_ke), orang - 1)
                temp_keterangan.append(hasil_keterangan)
                temp_img.append(filename_img)
                if hasil_keterangan == 'benar':
                    temp_benar+=1
                else:
                    temp_salah+=1

            hasil_akurasi=(temp_benar/(temp_benar+temp_salah))*100
            return render_template('LDA.html',query_path="SEMUA",len_data_test=ORL_face.data_test.shape[0],img=temp_img,akurasi=hasil_akurasi,keterangan=temp_keterangan)
    else:
        return render_template('LDA.html')




if __name__ == "__main__":
    app.run(host="127.0.0.1")
