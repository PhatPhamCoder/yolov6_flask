# from flask import Flask, render_template, request
# import os
# from random import random
# from my_yolo import my_yolo
# import cv2
# from utils import *

# yolov_model = my_yolo("weights/mass_detect.pt","cpu","data/data.yaml", 640, True)

# # Khởi tạo Flask
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = "static"
# app.config['ALLOWED_EXTENSIONS'] = ['.dicom']
# result_folder = "/result_images"

# # Hàm xử lý request
# @app.route("/", methods=['GET', 'POST'])
# def home_page():
#     # Nếu là POST (gửi file)
#     if request.method == "POST":
#         try:        
#             if not os.path.exists('static'):
#                 os.mkdir('static')

#             # Handle DICOM file uploads
#             dicom_list = []
#             for file in request.files.getlist('file'):
#                 filename = file.filename
#                 file_ext = os.path.splitext(filename)[1].lower()
#                 if file_ext in app.config['ALLOWED_EXTENSIONS']:
#                     dicom_list.append(filename)
#                     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


#             output_dir = os.path.join(app.config['UPLOAD_FOLDER'])
#             # Convert DICOM images to PNG
#             image_size_list = convert_dicom_to_png(dicom_list,output_dir)

#             print("file.filename::",file.filename)
#             print(image_size_list)
#             filename = file.filename[:-6] + ".jpg"
#             print(image_size_list)
#             if filename in image_size_list:
#                 index = image_size_list.index(filename)
#                 print(f"Tìm thấy file {filename} tại vị trí index = {index}")
#                 print(filename[index])
#                 image = image_size_list[index]

#                 if image:
#                     path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image)
#                     print(path_to_save)
#                     image.save(path_to_save)

#                     frame = cv2.imread(path_to_save)
#                     frame, ndet = yolov_model.infer(frame, conf_thres=0.6, iou_thres=0.45)
#                     if ndet!=0:
#                         cv2.imwrite(path_to_save, frame)
                    
#                         # Trả về kết quả
#                         print("Tải file lên thành công")
#                         return render_template("index.html", user_image = image , rand = str(random()),
#                                             msg="Tải file lên thành công", ndet = ndet)

#                     else:
#                         print("Không nhận diện được vật thể")
#                         return render_template("index.html", user_image = image , rand = str(random()),
#                                             msg="Không nhận diện được vật thể", ndet = ndet)
#                 else:
#                     # Nếu không có file thì yêu cầu tải file
#                     return render_template('index.html', msg='Hãy chọn file để tải lên')
#             else:
#                 return render_template('index.html', msg='Đã xảy ra lỗi vui lòng thử lại')


#         except Exception as ex:
#             # Nếu lỗi thì thông báo
#             print(ex)
#             return render_template('index.html', msg='Không nhận diện được vật thể')

#     else:
#         # Nếu là GET thì hiển thị giao diện upload
#         return render_template('index.html')


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, port=8080)

from flask import Flask, render_template, request
import os
from random import random
from my_yolo import my_yolo
import cv2
from utils import convert_dicom_to_png

yolov_model = my_yolo("weights/mass_detect.pt", "cpu", "data/data.yaml", 640, True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
app.config['ALLOWED_EXTENSIONS'] = ['.dicom']
result_folder = "/result_images"

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        try:
            if not os.path.exists('static'):
                os.mkdir('static')

            dicom_list = []
            files = request.files.getlist('file')
            file_names = [file.filename for file in files]
            results = []

            for file in files:
                filename = file.filename
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in app.config['ALLOWED_EXTENSIONS']:
                    dicom_list.append(filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            output_dir = os.path.join(app.config['UPLOAD_FOLDER'])
            image_size_list = convert_dicom_to_png(dicom_list, output_dir)
            jpg_files = [filename for filename in image_size_list if filename.endswith('.jpg')]

            list1_names = [os.path.splitext(filename)[0] for filename in file_names]
            list2_names = [os.path.splitext(filename)[0] for filename in jpg_files]
            common_elements = set(list1_names).intersection(set(list2_names))
            jpg_names = [filename + ".jpg" for filename in common_elements]

            for filename in jpg_names:
                if filename.endswith(".jpg"):
                    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                    frame = cv2.imread(path_to_save)
                    try:
                        frame, ndet = yolov_model.infer(frame, conf_thres=0.6, iou_thres=0.45)

                        results.append({
                            'filename': filename,
                            'objects': frame,
                            'ndet': ndet
                        })

                        if ndet != 0:
                            cv2.imwrite(path_to_save, frame)
                            print(f"Nhân diện {filename} thành công")
                        else:
                            print(f"Không nhận diện được vật thể trong file {filename}")

                    except Exception as ex:
                        print(f"Error during inference for file {filename}: {ex}")
                        # Log the error or handle it appropriately

            # Move return statement outside the loop to process all images
            return render_template("index.html", results=results)

        except Exception as ex:
            print(f"Error: {ex}")
            return render_template('index.html', msg='Không nhận diện được vật thể hoặc có lỗi xảy ra')

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)


