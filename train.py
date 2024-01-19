from tools import getDataset

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim

from PIL import Image

from model import StyleTransferUNet
from dataset import StyleTransferDataset

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS device not found.")
print('Device:', device)

# Training loop
def train_style_transfer_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=20, device='cuda'):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i_batch, (images, targets, labels) in enumerate(train_dataloader):

            content_image = images.to(device)
            style_image = targets.to(device) 

            optimizer.zero_grad()
            generated_image = model(content_image)
            loss = criterion(generated_image, style_image)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}')

        # Validation
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for i_batch, (images, targets, labels) in enumerate(val_dataloader):
                content_image_val = images.to(device)
                style_image_val = targets.to(device) 
                generated_image_val = model(content_image_val)
                val_loss = criterion(generated_image_val, style_image_val)
                validation_loss += val_loss.item()

            average_validation_loss = validation_loss / len(val_dataloader)
            print(f'Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {average_validation_loss}')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model/style_transfer_model.pth')

def test_style_transfer_model(model, test_dataloader, criterion, device='cuda'):
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_loss = 0.0
        for i_batch, (images, targets, labels) in enumerate(test_dataloader):
            content_image_test = images.to(device)
            style_image_test = targets.to(device) 
            generated_image_test = model(content_image_test)
            test_loss += criterion(generated_image_test, style_image_test).item()

        average_test_loss = test_loss / len(test_dataloader)
        print(f'Test - Loss: {average_test_loss}')


def main ():
    # Load dataset
    root_dir = os.path.join(os.getcwd(), 'VD_dataset2')
    dataset = getDataset(path=root_dir, shuffle_images=False)

    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

    # Split dataset into train and test
    train_set, temp_set = train_test_split(dataset, test_size=0.3, random_state=42)
    val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

    train_dataset = StyleTransferDataset(train_set, transform=transform)
    val_dataset = StyleTransferDataset(val_set, transform=transform)
    test_dataset = StyleTransferDataset(test_set, transform=transform)

    batch_size = 64
    num_workers = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create model, loss function and optimizer
    style_transfer_model = StyleTransferUNet().to(device)

    criterion = BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(style_transfer_model.parameters(), lr=1e-3)

    train_style_transfer_model(style_transfer_model, train_loader, validation_loader, criterion, optimizer, num_epochs=20, device=device)
    test_style_transfer_model(style_transfer_model, test_loader, criterion, device=device)

if __name__ == "__main__":
    main()