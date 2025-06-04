import torch
from torchvision import transforms
from PIL import Image
import os

def predict_video(model, folder_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    face_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                face_images.append(transform(img).unsqueeze(0))
            except:
                continue

    if not face_images:
        return -1, 0.0  # No valid faces

    inputs = torch.cat(face_images)
    outputs = model(inputs)
    softmax = torch.nn.functional.softmax(outputs, dim=1)
    avg_probs = softmax.mean(dim=0)

    predicted_class = torch.argmax(avg_probs).item()
    confidence = avg_probs[predicted_class].item() * 100
    return predicted_class, round(confidence, 2)

def evaluate_model(model, dataset):
    from torch.utils.data import DataLoader
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=16):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")
