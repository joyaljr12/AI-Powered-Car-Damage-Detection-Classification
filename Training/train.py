import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import car_classifier_resnet
from dataset import get_dataloader

# Load training data
train_loader,_ = get_dataloader(Batch_size=32)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the appropriate device
model = car_classifier_resnet(num_classes= 6).to(device)

# Define loss function (CrossEntropyLoss is used for multi-class classification)
loss_function = nn.CrossEntropyLoss()

# Define optimizer (Adam optimizer with learning rate 0.001 and weight decay for regularization)
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.004)

# Training Loop function
def train_model(epochs = 5):
    model.train()
    start = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_num, (images, labels) in enumerate(train_loader): # Loop through batches
            images = images.to(device) # Move images to GPU/CPU
            labels = labels.to(device) # Move labels to GPU/CPU

            #forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels) # Compute loss

            # Zero the gradients to prevent accumulation
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimization step: update model parameters
            optimizer.step()

            # Print loss for every 10 batches
            if (batch_num + 1) % 10 == 0:
                print(f'Batch:{batch_num+1}, Epoch :{epoch+1}, Loss: {loss.item():0.2f}')

            running_loss += loss.item() # Accumulate total loss for the epoch
        
        # Compute and print average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg loss: {epoch_loss:.4f}")

    end = time.time() # End timing the training process
    print(f'Training completed in {end - start:.2f} seconds')

    # Save trained model
    torch.save(model.state_dict(), "models/trained_model.pth")
    print("Model saved successfully!")

# Run training if script is executed directly
if __name__ == "__main__":
    train_model()

       