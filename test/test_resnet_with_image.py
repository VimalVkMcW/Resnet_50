import torch
import sys
from compare_pcc import PCC
sys.path.append("../")
from torchvision.models.resnet import resnet50
from Resnet.Resnet import resnet
from torchvision import transforms
from PIL import Image
import requests


def load_model(model_path):
    model = resnet(model_path)
    return model

def validate_accuracy(model, img_path, device):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    input_batch = input_batch.to(device)
    model.to(device)
    
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    _, indices = torch.topk(probabilities, 5)
    percentage = probabilities[indices].tolist()
    
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(LABELS_URL)
    labels = response.text.split("\n")
    
    for i in range(5):
        print(f"{labels[indices[i]]}: {percentage[i]*100:.2f}%")
    
    with open("../output/torchvision_output.txt", "w") as f:
        f.write(str(output))

    return probabilities, indices


model_path = '../dataset/resnet50-0676ba61.pth'
img_path = '../dataset/fish.jpg'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(model_path)
probabilities, indices = validate_accuracy(model, img_path, device)