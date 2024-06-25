import os
import numpy as np
import cv2
import torch
import torch.utils.data
import scipy.io
from core import model
import argparse
from mtcnn import MTCNN

def trainSpecificPerson(face_data_dir, required_shape=(112, 112), model_path='./model/best/068.ckpt', save_path='./result/specific_person_features.mat'):
    # Initialize MobileFaceNet model
    net = model.MobileFacenet()

    # Load pre-trained model
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # GPU 없이 CPU에서 로드
        net.load_state_dict(checkpoint['net_state_dict'])
    else:
        raise FileNotFoundError(f"Cannot find model checkpoint at {model_path}")

    net.eval()

    # Face detector
    face_detector =  MTCNN()# Implement or use your face detection method here

    # Prepare for feature extraction
    encodes = []
    encoding_dict = {}

    # Iterate through each person's directory
    for person_name in os.listdir(face_data_dir):
        person_dir = os.path.join(face_data_dir, person_name)
        encodes = []  # Initialize encodes for each person

        # Iterate through each image in the person's directory
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            # Load and preprocess image
            img_BGR = cv2.imread(image_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            # Detect face using face_detector (modify according to your setup)
            x = face_detector.detect_faces(img_RGB)
            if x:  # Face detected
                x1, y1, width, height = x[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img_RGB[y1:y2, x1:x2]

                # Preprocess face
                face = (face - 127.5) / 128.0  # Normalize
                face = cv2.resize(face, required_shape)

                # Convert to tensor and move to CPU
                face_tensor = torch.tensor(face, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
                
                # Extract feature using MobileFaceNet
                with torch.no_grad():
                    encode = net(face_tensor).numpy()

                encodes.append(encode)

        # Calculate and store average encoding for the person
        if encodes:
            encode = np.mean(encodes, axis=0)
            encoding_dict[person_name] = encode

    # Save encodings to a .mat file
    scipy.io.savemat(save_path, encoding_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training specific person')
    parser.add_argument('--face_data_dir', type=str, default='./faces/', help='Directory containing face images')
    parser.add_argument('--model_path', type=str, default='./model/best/068.ckpt', help='Path to pretrained MobileFaceNet model checkpoint')
    parser.add_argument('--save_path', type=str, default='./result/specific_person_features.mat', help='Path to save extracted features (.mat file)')
    args = parser.parse_args()

    # Train specific person and save features
    trainSpecificPerson(args.face_data_dir, model_path=args.model_path, save_path=args.save_path)
