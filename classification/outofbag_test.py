from datasets import load_dataset
import pandas as pd
import torch
from torchvision import models, transforms
import torch.nn as nn


classes = ["letter", "form", "email", "handwritten", "advertisement", "scientific report",
           "scientific publication", "specification", "file folder", "news article", "budget",
           "invoice", "presentation", "questionnaire", "resume", "memo"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    try:
        model.load_state_dict(torch.load('model.pth', map_location=device), strict=False)
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        return None

model = load_model()

ds = load_dataset("amaye15/invoices-google-ocr")
df = pd.DataFrame(ds['test'])
# pixel_values 	label 	ocr
# <PIL.PngImagePlugin.PngImageFile image mode=RG...
N = 50
random_images= df.sample(N)
random_images = random_images['pixel_values']
true = ['invoice'] * N
resulted = []
for image in random_images:
    # image channels 1
    image= image.convert("RGB")
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print(classes[predicted.item()])
        resulted.append(classes[predicted.item()])

# compute accuracy
accuracy = sum([1 for i, j in zip(true, resulted) if i == j]) / N
print(f"Accuracy: {accuracy}")

# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(true, resulted, labels=classes)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
