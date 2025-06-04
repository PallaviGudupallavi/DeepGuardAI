from facenet_pytorch import MTCNN
import cv2
import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()


def extract_faces_from_video(video_path, output_dir, max_frames=10):
    mtcnn = MTCNN(keep_all=False)
    cap = cv2.VideoCapture(video_path)
    frames_extracted = 0

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened() and frames_extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        face = mtcnn(img)
        if face is not None:
            img_pil = to_pil(face)
            save_path = os.path.join(output_dir, f"frame{frames_extracted}.jpg")
            img_pil.save(save_path)

    cap.release()
