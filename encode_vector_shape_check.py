# import cv2 
# import numpy as np
# from core.model import MobileFacenet
# from scipy.spatial.distance import cosine
# import torch
# import scipy.io
# import pickle
# import pathlib

# def normalize(img):
#     return (img - 127.5) / 128.0

# def get_encode(face_encoder, face, size):
#     face = normalize(face)
#     face = cv2.resize(face, size)
    
#     # Convert face to tensor and move to device
#     face_tensor = torch.from_numpy(face.transpose((2, 0, 1))).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    
#     # Move face_encoder to the same device as face_tensor
#     face_encoder = face_encoder.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
#     # Forward pass through face_encoder
#     with torch.no_grad():
#         encode = face_encoder(face_tensor)
#     print(encode.shape)
#     # Move encode to CPU and convert to numpy array
#     encode = encode.cpu().numpy()[0]  # Convert tensor to numpy array and remove batch dimension
    
#     return encode



# if __name__ == '__main__':
#     # Load MobileFaceNet model
#     face_encoder = MobileFacenet()
#     face_encoder.eval()  # Set model to evaluation mode
    
#     # Load an example image
#     image_path = 'C:/GRADU/face_recognition/faces/Junghyun/KakaoTalk_20240524_000815061.jpg'  # Replace with your image path
#     face_image = cv2.imread(image_path)
#     face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
#     # Define the required size for face embedding
#     required_size = (112, 112)
    
#     # Get face encoding
#     face_encoding = get_encode(face_encoder, face_image, required_size)
    
#     # Print shape of face encoding
#     print("Shape of face encoding:", face_encoding.shape)


# import cv2
# import numpy as np
# import torch
# from core.model import MobileFacenet

# def normalize(img):
#     return (img - 127.5) / 128.0

# def get_encode(face_encoder, face, size):
#     face = normalize(face)
#     face = cv2.resize(face, size)
    
#     # Convert face to tensor and move to device
#     face_tensor = torch.from_numpy(face.transpose((2, 0, 1))).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    
#     # Move face_encoder to the same device as face_tensor
#     face_encoder = face_encoder.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
#     # Forward pass through face_encoder
#     with torch.no_grad():
#         encode = face_encoder(face_tensor)
    
#     # Move encode to CPU and convert to numpy array
#     encode = encode.cpu().numpy()[0]  # Convert tensor to numpy array and remove batch dimension
    
#     return encode

# if __name__ == '__main__':
#     # Load MobileFaceNet model
#     face_encoder = MobileFacenet()
#     face_encoder.eval()  # Set model to evaluation mode
    
#     # Define the required size for face embedding
#     required_size = (112, 112)
    
#     # Open the webcam
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Failed to open webcam.")
#         exit()
    
#     while True:
#         # Read frame from webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame from webcam.")
#             break
        
#         # Convert frame to RGB (MobileFacenet model expects RGB input)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Perform face detection here to get face bounding box(es)
#         # For simplicity, let's assume you already have the face bounding box in a variable `face_box`
#         # You can use face detection models like Haar cascades or deep learning based detectors
        
#         # Example bounding box coordinates (replace with actual face detection)
#         # For testing, let's assume the entire frame is the face box
#         face_box = (0, 0, frame.shape[1], frame.shape[0])
        
#         # Extract the face region from the frame
#         x1, y1, x2, y2 = face_box
#         face_img = rgb_frame[y1:y2, x1:x2]
        
#         # Get face encoding
#         face_encoding = get_encode(face_encoder, face_img, required_size)
        
#         # Print shape of face encoding
#         print("Shape of face encoding:", face_encoding.shape)
        
#         # Display the frame with bounding box and label
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.imshow('Webcam', frame)
        
#         # Exit the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

import cv2
import numpy as np
import torch
from core.model import MobileFacenet

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

if __name__ == '__main__':
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
    
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open webcam.")
        exit()
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        
        # Convert frame to RGB (MobileFacenet model expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform face detection here to get face bounding box(es)
        # For simplicity, let's assume you already have the face bounding box in a variable `face_box`
        # You can use face detection models like Haar cascades or deep learning based detectors
        
        # Example bounding box coordinates (replace with actual face detection)
        # For testing, let's assume the entire frame is the face box
        face_box = (0, 0, frame.shape[1], frame.shape[0])
        
        # Extract the face region from the frame
        x1, y1, x2, y2 = face_box
        face_img = rgb_frame[y1:y2, x1:x2]
        
        # Get face encoding
        face_encoding = get_encode(face_encoder, face_img, required_size)
        
        # Print shape of face encoding
        print("Shape of face encoding:", face_encoding.shape)
        
        # Display the frame with bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)
        
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
