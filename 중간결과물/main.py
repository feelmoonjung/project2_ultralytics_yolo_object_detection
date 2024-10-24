from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pathlib import Path
import imageio
import cv2
# from tempfile import TemporaryFile
import tempfile
import numpy as np

app = Flask(__name__)
model = YOLO('C:\Users\jungf\runs\detect\train3\weights\best.pt')

@app.route('/')
def home():
    return 'This is Home!'

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기
        file = request.files['image']
        file.save('static/uploads/' + secure_filename(file.filename))
        if not file:
            return render_template('index.html', ml_label = "No Files")
        with tempfile.TemporaryDirectory() as td:
            temp_filename = Path(td) / 'uploads'
            file.save(temp_filename)
            filename = str(file)
            pathname = 'uploads/' + filename.split(' ')[1][1:-1]
        img = imageio.imread(file)

        imgCropped = img[0:800,0:800] 
        imgResize = cv2.resize(imgCropped,(5000,5000)) 
        img = imgResize
        # Convert the image to RGB
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # We resize the image to the input width and height of the first layer of the network.    
        resized_image = cv2.resize(original_image, (m.width, m.height))

        bgrLower = np.array([0, 0, 0])    
        bgrUpper = np.array([100, 100, 100])    
        img_mask = cv2.inRange(resized_image, bgrLower, bgrUpper) 
        black = invert(img_mask)
        resized_image[black>0]=(255,255,255)

        # 3. RUN OBJECT DETECTION MODEL

        # Set the IOU threshold. Default value is 0.4
        iou_thresh = 0.5

        # Set the NMS threshold. Default value is 0.6
        nms_thresh = 0.5

        # Detect objects in the image
        boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)

        # Print the objects found and the confidence level
        obj = objects_info(boxes, class_names)

        #Plot the image with bounding boxes and corresponding object class labels
        plot_boxes(original_image, boxes, class_names, plot_labels = True)

        return render_template('predict.html', ml_label=obj, print_image = pathname )

if __name__ == '__main__' :
    app.run(host = '0.0.0.0', port=8000, debug = True)