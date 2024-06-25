import cv2
import numpy as np
import mtcnn
from core.model import MobileFacenet
import torch
import scipy.io
from scipy.spatial.distance import cosine
import time

confidence_t = 0.99
recognition_t = 0.5
required_size = (112, 112)  # MobileFacenet의 입력 크기에 맞춰 변경

def normalize(img):
    return (img - 127.5) / 128.0

def l2_normalizer(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        encode = face_encoder(face_tensor).numpy()
    encode = l2_normalizer(encode)
    return encode

def load_matfile(matfile_path):
    encoding_dict = {}
    mat_data = scipy.io.loadmat(matfile_path)
    for key in mat_data:
        if isinstance(mat_data[key], np.ndarray) and mat_data[key].shape[0] > 0:
            encoding_dict[key] = mat_data[key][0]
    return encoding_dict

def detect(img, detector, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            db_encode = db_encode.flatten()  # 1차원 벡터로 변환
            encode = encode.flatten()  # 1차원 벡터로 변환
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, f'{name}__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img

if __name__ == "__main__":
    face_encoder = MobileFacenet()
    model_path = './model/best/068.ckpt'  # 학습된 모델 경로
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    face_encoder.load_state_dict(checkpoint['net_state_dict'])
    face_encoder.eval()

    encodings_path = './result/specific_person_features.mat'  # 저장된 인코딩 경로
    encoding_dict = load_matfile(encodings_path)

    face_detector = mtcnn.MTCNN()

    cap = cv2.VideoCapture(0)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # FPS 계산을 위한 초기화
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break
        
        # Write the frame to the file
        out.write(frame)

        frame = detect(frame, face_detector, face_encoder, encoding_dict)

        # FPS 계산
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
