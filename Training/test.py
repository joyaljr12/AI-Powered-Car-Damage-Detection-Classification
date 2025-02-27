import torch
import torch.nn as nn
import time
from dataset import get_dataloader
from model import car_classifier_resnet



# Load test data
_, test_loader = get_dataloader(Batch_size=32)

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model = car_classifier_resnet(num_classes=6).to(device)  # Ensure model is initialized correctly
model.load_state_dict(torch.load("models/trained_model.pth"))  # Load saved weights
model.eval()  # Set model to evaluation mode


# Validation
def test_model():
    """Function to evaluate the model on test data"""
    start = time.time()
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No gradients needed during testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get predicted class

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    end = time.time()
    
    print(f'Validation Accuracy: {accuracy:.2f}%')
    print(f'Execution Time: {end - start:.2f} seconds')

    return all_labels, all_predictions

# Run the test function
if __name__ == "__main__":
    all_labels, all_predictions = test_model()

