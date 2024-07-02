import cv2 
import numpy as np
from core.model import MobileFacenet
from scipy.spatial.distance import cosine
import torch
import scipy.io
import pickle
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 임계값 및 설정값 정의
confidence_t = 0.99  # 얼굴 검출 확률 임계값. Yolo v5가 얼굴로 확신하는 경우만 검출하도록 한다.
recognition_t = 0.5   # 인식 임계값. 인식된 인물과의 코사인 유사도가 이 임계값 이상일 경우만 인식하도록 한다.
required_size = (160, 160)

def load_matfile(matfile_path):
    encoding_dict = {}
    mat_data = scipy.io.loadmat(matfile_path)
    for key in mat_data:
        if isinstance(mat_data[key], np.ndarray) and mat_data[key].shape[0] > 0:
            encoding_dict[key] = mat_data[key][0]
    return encoding_dict

def normalize(img):
    return (img - 127.5) / 128.0

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def apply_mosaic(image, pt_1, pt_2, kernel_size=15):
    x1, y1 = pt_1
    x2, y2 = pt_2
    face_height, face_width, _ = image[y1:y2, x1:x2].shape
    face = image[y1:y1+face_height, x1:x1+face_width]
    face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
    face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
    image[y1:y1+face_height, x1:x1+face_width] = face
    return image

def run_yolo(frame, model, face_encoder, encoding_dict):
    results = model(frame)
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        label = f'{model.names[int(cls)]} {conf:.2f}'
        
        if conf >= confidence_t:
            if model.names[int(cls)] == 'face':
                # 얼굴 부분 추출
                face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                # 추출한 얼굴을 FaceNet 모델에 입력하여 인코딩
                encode = get_encode(face_encoder, face_img, required_size)
                
                # 얼굴 인식
                name = 'unknown'
                distance = float("inf")
                for db_name, db_encode in encoding_dict.items():
                    dist = cosine(db_encode, encode)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist
                
                # 결과 표시 및 모자이크 적용
                if name == 'unknown':
                    frame = apply_mosaic(frame, (int(x1), int(y1)), (int(x2), int(y2)))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # 다른 클래스의 객체는 그대로 표시하되, 모자이크 적용
                frame = apply_mosaic(frame, (int(x1), int(y1)), (int(x2), int(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

if __name__ == "__main__":
    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./627.pt', force_reload=True)
    
    # Load FaceNet model
    face_encoder = MobileFacenet()
    model_path = './model/best/068.ckpt'  # 학습된 모델 경로
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    face_encoder.load_state_dict(checkpoint['net_state_dict'])
    face_encoder.eval()

    # Load encoding dictionary
    encodings_path = './result/specific_person_features.mat'  # 저장된 인코딩 경로
    encoding_dict = load_matfile(encodings_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_combined1.avi', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("CAM NOT OPENED")
            break

        # YOLO detection 및 얼굴 인식
        frame = run_yolo(frame, model, face_encoder, encoding_dict)
        
        out.write(frame)
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
