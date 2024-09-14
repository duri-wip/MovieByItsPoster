import torch
from PIL import Image

def predict(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  
    
    with torch.no_grad():
        output = model(image)
    
    return output.item()