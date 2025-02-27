from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models


trained_model = None
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

class car_classifier_resnet(nn.Module): 
    def __init__(self, num_classes):  
        super().__init__()
        self.model_res = models.resnet50(weights='DEFAULT')

        # Freeze all layers except the final fully connected layer
        for param in self.model_res.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 and fc layers
        for param in self.model_res.layer4.parameters():
            param.requires_grad = True  

        in_features = self.model_res.fc.in_features

        self.model_res.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)

        )
    
    def forward(self, x):
        return self.model_res(x)
    
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
    
    
        trained_model = car_classifier_resnet(num_classes=6)
        trained_model.load_state_dict(torch.load("models/trained_model.pth")) # map_location=torch.device("cpu")
        trained_model.eval()
   

    with torch.no_grad():
        output = trained_model(image_tensor) 
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]



    