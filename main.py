import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from src.Dataset import PosterDataset
from src.Model import ResnetModel
from src.Train import train_model
from src.Predict import predict

transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                ])

dataset = PosterDataset(image_dir = '../poster_downloads', transform = transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ResnetModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, dataloader, criterion, optimizer)


image_path = "QuietPlace3.jpg"
predicted_rating = predict(model, image_path, transform)
print(f'Predicted rating: {predicted_rating:.2f}')
