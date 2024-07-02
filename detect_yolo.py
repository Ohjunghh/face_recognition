import cv2
import numpy as np
from core.model import MobileFacenet
from scipy.spatial.distance import cosine
import torch
import scipy.io
import pickle


# Thresholds and settings
confidence_t = 0.80  # Object detection confidence threshold
recognition_t = 0.5  # Face recognition threshold
required_size = (160, 160)  # Required face image size

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
    
    # Convert face to tensor and move to device
    face_tensor = torch.from_numpy(face.transpose((2, 0, 1))).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move face_encoder to the same device as face_tensor
    face_encoder = face_encoder.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Forward pass through face_encoder
    with torch.no_grad():
        encode = face_encoder(face_tensor)
    
    # Move encode to CPU and convert to numpy array
    encode = encode.cpu().numpy()[0]  # Convert tensor to numpy array and remove batch dimension
    
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
    results = model(frame)  # Perform object detection
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        label = f'{model.names[int(cls)]} {conf:.2f}'
        
        if conf >= confidence_t:
            # Extract object region
            object_img = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Check if the detected object is a face
            if model.names[int(cls)] == 'face':
                # Encode face using FaceNet model
                #face_img = cv2.resize(object_img, required_size)
                encode_face = get_encode(face_encoder, object_img, required_size)  # Rename to avoid overwrite
                

                # Face recognition
                name = 'unknown'
                distance = float("inf")
                for db_name, db_encode in encoding_dict.items():
                    # Ensure both encode_face and db_encode are numpy arrays with the same shape
                    encode = np.array(encode_face)
                    db_encode = np.array(db_encode)
                    
                    # Normalize vectors
                    encode = encode / np.linalg.norm(encode)
                    db_encode = db_encode / np.linalg.norm(db_encode)
                    
                    # Check dimensions of encode and db_encode
                    if encode.shape != db_encode.shape:
                        print("encode.shape != db_encode.shape")
                        print("encode.shape:"+encode.shape)
                        print("db_encode.shape:"+db_encode.shape)
                        continue  # Skip if dimensions don't match

                    # Calculate cosine distance
                    dist = cosine(encode, db_encode)
                    #print("dist:",dist)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist
                
                # Display results
                if name == 'unknown':
                    frame = apply_mosaic(frame, (int(x1), int(y1)), (int(x2), int(y2)))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            else:
                # Apply mosaic to non-face objects and display labels
                frame = apply_mosaic(frame, (int(x1), int(y1)), (int(x2), int(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

if __name__ == "__main__":
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath
    
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./627.pt', force_reload=True)
    print("YOLOv5 model loaded successfully.")
    
    # YOLOv5 설정
    model.conf = 0.5  # Detection confidence threshold
    model.classes = None  # 모든 클래스 사용
    model.agnostic_nms = False  # 클래스 독립적인 NMS 설정
    
    # Define the path to the model checkpoint
    model_path = 'C:/GRADU/face_recognition/model/best/068.ckpt'
    
    # Load MobileFaceNet model
    face_encoder = MobileFacenet()
    face_encoder.eval()  # Set model to evaluation mode
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load state_dict into MobileFacenet model
    face_encoder.load_state_dict(checkpoint['net_state_dict'])
    
    # Define the required size for face embedding
    required_size = (112, 112)
    

    # 얼굴 인코딩 데이터 로드
    encodings_path = './result/specific_person_features.mat'
    encoding_dict = load_matfile(encodings_path)

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # 비디오 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_combined1.avi', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # YOLOv5을 사용한 객체 검출 및 얼굴 인식 실행
        frame = run_yolo(frame, model, face_encoder, encoding_dict)

        # 비디오 파일에 프레임 추가
        out.write(frame)

        # 프레임을 창에 표시
        cv2.imshow('Camera', frame)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
