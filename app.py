from flask import Flask, render_template, request
import os
from random import random
from my_yolo import my_yolo
import cv2
from utils import convert_dicom_to_png
from pymongo import MongoClient
import pandas as pd

yolov_model = my_yolo("weights/mass_detect.pt", "cpu", "data/data.yaml", 640, True)

# Read CSV file using pandas
csv_file_path = "data.csv"
df = pd.read_csv(csv_file_path)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
app.config['ALLOWED_EXTENSIONS'] = ['.dicom']

@app.route("/", methods=['GET', 'POST'])

def home_page():
    if request.method == "POST":
        try:
            if not os.path.exists('static'): # Kiểm tra folder lưu ảnh upload có tồn tại hay không
                os.mkdir('static') # Tạo folder lưu ảnh nếu chưa có

            # Set up MongoDB connection 
            with MongoClient('mongodb://localhost:27017') as client: #Kết nối với Database mongoDB
                db = client['DetectMassProgram']
                collection = db['data']

                dicom_list = [] # Danh sách các file dicom đã upload
                files = request.files.getlist('file') # Danh sách các file dicom đã upload

                file_names = [file.filename for file in files]   # Danh sách các tên của các file dicom đã upload
                results = [] # Danh sách các kết quả được xử lý

                for file in files: # Lặp qua các file dicom đã upload
                    filename = file.filename # Tên của file dicom đã upload
                    file_ext = os.path.splitext(filename)[1].lower() # Định dạng của file dicom đã upload
                    if file_ext in app.config['ALLOWED_EXTENSIONS']: # Kiểm tra định dạng của file dicom đã upload có hợp lệ hay không
                        dicom_list.append(filename) # Thêm tên của file dicom đã upload vào danh sách các file dicom đã upload
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # Lưu file dicom đã upload vào thư mục static

                output_dir = os.path.join(app.config['UPLOAD_FOLDER']) # Đường dẫn đến thư mục lưu ảnh JPG
                image_size_list = convert_dicom_to_png(dicom_list, output_dir) # Chuyển đổi các file dicom đã upload thành ảnh JPG
                jpg_files = [filename for filename in image_size_list if filename.endswith('.jpg')] # Danh sách các file ảnh JPG

                list1_names = [os.path.splitext(filename)[0] for filename in file_names] # Danh sách các tên của các file dicom đã upload

                list2_names = [os.path.splitext(filename)[0] for filename in jpg_files] # Danh sách các tên của các file ảnh JPG

                common_elements = set(list1_names).intersection(set(list2_names)) # Danh sách các tên của các file dicom và file ảnh JPG có trong nhau

                jpg_names = [filename + ".jpg" for filename in common_elements] # Danh sách các tên của các file ảnh JPG

                resultCC = [element for element in jpg_names if 'CC' in element] # Danh sách các tên của các file ảnh JPG có trong tên có CC
                resultMLO = [element for element in jpg_names if 'MLO' in element] # Danh sách các tên của các file ảnh JPG có trong tên có MLO

                def custom_sort(filename): # Hàm sắp xếp các file ảnh JPG theo tên
                    is_left = 'R' in filename # Kiểm tra xem tên có chứa chữ R hay không
                    return (is_left, filename) # Trả về (True, tên) nếu tên có chứa chữ R, ngược lại (False, tên)

                resultCC.sort(key=custom_sort, reverse=True) # Sắp xếp các file ảnh JPG theo tên
                resultMLO.sort(key=custom_sort, reverse=True) # Sắp xếp các file ảnh JPG theo tên

                list_name = resultCC + resultMLO # Danh sách các tên của các file ảnh JPG

                additional_info = request.form['additional_info'] # Thông tin thêm về bệnh nhân

                data = [{'case': additional_info, 'filename': name, 'data': 'data_for_' + name,
                         'position': os.path.splitext(name)[0]} for name in list_name] # Dữ liệu để lưu vào MongoDB

                collection.insert_many(data) # Lưu dữ liệu vào MongoDB

                for filename in list_name: # Lặp qua các tên của các file ảnh JPG
                    if filename.endswith(".jpg"): # Kiểm tra xem tên của file ảnh JPG có đuôi là .jpg không
                        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Đường dẫn đến file ảnh JPG
                        frame = cv2.imread(path_to_save) # Đọc file ảnh JPG vào biến frame
                        
                        try:
                            frame, ndet = yolov_model.infer(frame, conf_thres=0.6, iou_thres=0.45) # Nhân diện ảnh JPG và lấy kết quả nhân diện

                            results.append({
                                'filename': filename,
                                'ndet': ndet,
                                'position': os.path.splitext(filename)[0],
                            }) # Thêm kết quả nhân diện vào danh sách kết quả

                            if ndet != 0:
                                cv2.imwrite(path_to_save, frame)
                                print(f"Nhân diện {filename} thành công")

                                # Draw box check
                                img_row = df[df['name'] == filename] # Lấy dòng của file ảnh JPG trong DataFrame
                                if not img_row.empty: # Kiểm tra xem file có trong DataFrame không
                                    
                                    x_min = img_row['xmin'].values[0]
                                    x_max = img_row['xmax'].values[0]
                                    y_max = img_row['ymax'].values[0]
                                    y_min = img_row['ymin'].values[0]

                                    img = cv2.imread(path_to_save) # Đọc file ảnh JPG vào biến img

                                    box_width = x_max - x_min # Tính chiều rộng của khung hình
                                    box_height = y_max - y_min # Tính chiều cao của khung hình

                                    box_center_x = int(x_min + (box_width / 2)) # Tính tọa độ trung tâm của khung hình theo chiều ngang và chiều dọc của khung hình. Đây là t�
                                    box_center_y = int(y_min + (box_height / 2)) # Tính tọa độ trung tâm của khung hình theo chiều ngang và chiều dọc của khung hình. Đây là

                                    w = int(box_width) # Chuẩn hóa chiều rộng của khung hình
                                    h = int(box_height) # Chuẩn hóa chiều cao của khung hình

                                    x = box_center_x - int(w / 2) # Tính tọa độ x của khung hình theo chiều ngang và chiều dọc của khung hình. Đây là t�
                                    y = box_center_y - int(h / 2) # Tính tọa độ y của khung hình theo chiều ngang và chiều dọc của khung hình. Đây là t�

                                    img_with_box = img.copy() # Sao chép ảnh JPG vào biến img_with_box
                                    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 15) # Vẽ khung hình vào ảnh JPG và lấy kết quả vẽ khung hình vào biến img_with_box

                                    output_path = f"./static/{filename}" # Đường dẫn đến file ảnh JPG sau khi vẽ khung hình vào ảnh JPG
                                    cv2.imwrite(output_path, img_with_box) # Lưu ảnh JPG sau khi vẽ khung hình vào ảnh JPG vào thư mục static

                                    cv2.waitKey(0) # Chờ người dùng nhấn phím Enter để đóng ảnh JPG
                                else:
                                    print(f"Không tìm thấy file {filename} trong DataFrame.")
                            else:
                                print(f"Không nhận diện được vật thể trong file {filename}")

                        except Exception as ex:
                            print(f"Error during inference for file {filename}: {ex}")
            
                return render_template("result.html", results=results)

        except Exception as ex:
            print(f"Error: {ex}")
            return render_template('result.html', msg='Không nhận diện được vật thể hoặc có lỗi xảy ra')

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)

