import shutil
import pickle
from flask import Flask, render_template, request
import os
from random import random
from my_yolov6 import my_yolov6
import cv2
from utils import *\

yolov6_model = my_yolov6("weights/mass_detect.pt","cpu","data/data.yaml", 640, True)

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
app.config['ALLOWED_EXTENSIONS'] = ['.dicom']

# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
        try:
            # Create a new folder to store uploaded dicom files
            
            if not os.path.exists('static'):
                os.mkdir('static')

            # if not os.path.exists('static/result_images_dir'):
            #     os.mkdir('static/result_images_dir')

            # Handle DICOM file uploads
            dicom_list = []
            for file in request.files.getlist('file'):
                filename = file.filename
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in app.config['ALLOWED_EXTENSIONS']:
                    dicom_list.append(filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


            output_dir = os.path.join(app.config['UPLOAD_FOLDER'])
            # print("output_dir:::",output_dir)
            # # Convert DICOM images to PNG
            image_size_list = convert_dicom_to_png(dicom_list,output_dir)
            # print("image_size_list:::",image_size_list[1])

            image = image_size_list[1]
            # print(image)
            if image:
                # Lưu file

                # print("image.filename::::",image)
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image)
               
                # print("path_to_save:::",path_to_save)

                # image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)
                print(frame)
                frame, ndet = yolov6_model.infer(frame, conf_thres=0.6, iou_thres=0.45)
                print(ndet)
                if ndet!=0:
                    cv2.imwrite(path_to_save, frame)
                
                    # Trả về kết quả
                    print("Tải file lên thành công")
                    return render_template("index.html", user_image = image , rand = str(random()),
                                           msg="Tải file lên thành công", ndet = ndet)
                else:
                    print("Không nhận diện được vật thể")
                    return render_template("index.html", user_image = image , rand = str(random()),
                                           msg="Không nhận diện được vật thể", ndet = ndet)
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)