import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split


# Define data augmentation and normalization transformations
def get_dataloader(Batch_size = 32):
    image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(contrast=0.25, brightness= 0.25, saturation= 0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
    
    # Define dataset path (ensure 'dataset' folder contains images organized into subfolders per class)
    dataset_path = 'dataset'

    # Load dataset from the folder using ImageFolder (assumes images are structured in subdirectories by class)
    dataset = datasets.ImageFolder(root= dataset_path, transform= image_transform)

    # Split dataset into 75% training and 25% testing
    train_size = int(0.75*len(dataset))
    test_size = len(dataset) - train_size

    # Perform random split into training and testing datasets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle= True)
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle= False)

    return train_loader, test_loader # Return both training and testing DataLoaders