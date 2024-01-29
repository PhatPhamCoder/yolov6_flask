from flask import Flask, render_template, request
import os
from random import random
from my_yolo import my_yolo
import cv2
from utils import convert_dicom_to_png
from pymongo import MongoClient

yolov_model = my_yolo("weights/mass_detect.pt", "cpu", "data/data.yaml", 640, True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
app.config['ALLOWED_EXTENSIONS'] = ['.dicom']

@app.route("/", methods=['GET', 'POST'])

def home_page():
    if request.method == "POST":
        try:
            if not os.path.exists('static'):
                os.mkdir('static')

            # Set up MongoDB connection 
            with MongoClient('mongodb://localhost:27017') as client:
                db = client['DetectMassProgram']
                collection = db['data']

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

                resultCC = [element for element in jpg_names if 'CC' in element]
                # print(resultCC)
                resultMLO = [element for element in jpg_names if 'MLO' in element]
                # print(resultMLO)

                def custom_sort(filename):
                    is_left = 'R' in filename
                    return (is_left, filename)

                resultCC.sort(key=custom_sort, reverse=True)
                resultMLO.sort(key=custom_sort, reverse=True)

                print(resultCC)
                print(resultMLO)

                list_name = resultCC + resultMLO

                print(list_name)

                additional_info = request.form['additional_info']

                data = [{'case': additional_info, 'filename': name, 'data': 'data_for_' + name,
                         'position': os.path.splitext(name)[0]} for name in list_name]

                collection.insert_many(data)

                for filename in list_name:
                    if filename.endswith(".jpg"):
                        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                        frame = cv2.imread(path_to_save)

                        print(filename)
                        try:
                            frame, ndet = yolov_model.infer(frame, conf_thres=0.6, iou_thres=0.45)
                            print(additional_info)

                            results.append({
                                'filename': filename,
                                'ndet': ndet,
                                'position': os.path.splitext(filename)[0],
                            })

                            if ndet != 0:
                                cv2.imwrite(path_to_save, frame)
                                print(f"Nhân diện {filename} thành công")
                            else:
                                print(f"Không nhận diện được vật thể trong file {filename}")

                        except Exception as ex:
                            print(f"Error during inference for file {filename}: {ex}")
            
                print(results)
                return render_template("result.html", results=results)

        except Exception as ex:
            print(f"Error: {ex}")
            return render_template('result.html', msg='Không nhận diện được vật thể hoặc có lỗi xảy ra')

    else:
        return render_template('index.html')

if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True, port=8080)
    app.run(debug=True)

