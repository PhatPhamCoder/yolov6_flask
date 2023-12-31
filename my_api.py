from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import os
import my_yolo
import cv2

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

yolov6_model = my_yolo.my_yolov6("weights/mass_detect.pt", "cpu", "data/data.yaml", 640, False)

@app.route('/', methods=['POST'] )
def predict_yolov6():
    image = request.files['file']
    if image:
        # Lưu file
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        print("Save = ", path_to_save)
        image.save(path_to_save)

        frame = cv2.imread(path_to_save)

        # Nhận diên qua model Yolov6
        frame, no_object = yolov6_model.infer(frame)

        if no_object >0:
            cv2.imwrite(path_to_save, frame)

        del frame
        # Trả về đường dẫn tới file ảnh đã bounding box
        return path_to_save # http://server.com/static/path_to_save

    return 'Upload file to detect'

# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')