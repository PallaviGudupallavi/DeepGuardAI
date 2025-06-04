from facenet_pytorch import MTCNN
import cv2
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
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


class FaceDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.images = []
        self.label = label
        self.transform = transform

        # Walk through all subfolders
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.images.append(os.path.join(dirpath, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label
