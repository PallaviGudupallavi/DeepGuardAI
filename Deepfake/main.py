from dataset_loader import FaceDataset
from model import SimpleCNN
from train import train_model
from evaluate import evaluate_model
from torchvision import transforms
import torch

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    real_dataset = FaceDataset("dataset/real", label=0, transform=transform)
    fake_dataset = FaceDataset("dataset/fake", label=1, transform=transform)

    full_dataset = real_dataset + fake_dataset

    model = SimpleCNN()
    train_model(model, full_dataset, num_epochs=10)
    torch.save(model.state_dict(), 'model.pth')

    evaluate_model(model, full_dataset)
