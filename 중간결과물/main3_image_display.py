from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from IPython.display import display

# from tempfile import TemporaryFile

model = YOLO('C:\\Users\\jungf\\runs\\detect\\train3\\weights\\best.pt')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'file' not in request.files:
        return '파일이 전송되지 않았습니다.'
    
    file = request.files['file']
    if file.filename == '':
        return '선택한 파일이 없습니다.'
    
    # 모델에 예측 데이터를 넣을 때 가능한 확장자 데이터만 넣기(filter)
    ext = file.filename.split('.')[-1].lower()
    allowed_ext = {'jpg', 'jpeg', 'png', 'bmp'}

    # 예측 가능하게 설정
    if ext not in allowed_ext:
        return '허용되지 않은 확장자 입니다.'
    
    # 경로 정리 (사용자가 입력한 이미지 로컬 저장 > 해당 경로 변수 저장)
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok = True)
    file.save(file_path)

    result = model(file_path, save_txt = True)

    # 예측 결과를 추출하여 JSON 형태로 변환한 후 반환
    predictions = []
    for rs in result:
        # 클래스
        cls = rs.boxes.cls.numpy()
        # 확률값
        conf = rs.boxes.conf.numpy()
        # 좌표값
        box = rs.boxes.xyxy.numpy()

        for cls, conf, box in zip(cls, conf, box):
            pred = {
                'class' : int(cls), 
                'confidence' : float(conf),
                'x1' : float(box[0]), 
                'y1' : float(box[1]), 
                'x2' : float(box[2]), 
                'y2' : float(box[3])
            }
            predictions.append(pred)
            
        # return jsonify(predictions)
        return render_template('predict.html', ml_label = predictions)

if __name__ == '__main__' :
    app.run(host = '0.0.0.0', port=8000, debug = True)