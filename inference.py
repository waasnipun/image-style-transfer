import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import StyleTransferUNet
from tools import getDataset
import argparse

def apply_style_transfer(model, content_image, output_image_path, device):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    content_image = transform(content_image).unsqueeze(0)  # Add batch dimension

    content_image = content_image.to(device)

    with torch.no_grad():
        generated_image = model(content_image)

    generated_image = generated_image.cpu().squeeze().numpy()
    generated_image = (generated_image + 1) / 2.0

    generated_image = (generated_image * 255).astype('uint8')
    Image.fromarray(generated_image.transpose(1, 2, 0)).save(output_image_path)
    
    return Image.fromarray(generated_image.transpose(1, 2, 0))


def main(output_directory='output', image_path='VD_dataset2/0044_input.png'):

    model = StyleTransferUNet()
    model.load_state_dict(torch.load('trained_model/style_transfer_model.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    content_image = Image.open(image_path).convert('RGB')
    
    output_image_path = os.path.join(output_directory, f'generated_image.jpg')
    generated_image = apply_style_transfer(model, content_image, output_image_path, device)

    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot content image
    axes[0].imshow(content_image)
    axes[0].set_title('Content Image')
    axes[0].axis('off')

    # Plot generated image
    axes[1].imshow(generated_image)
    axes[1].set_title('Generated Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', type=str, default='output', help='Output directory for generated image')
    parser.add_argument('--image_path', type=str, default='', help='Path to the content image')
    args = parser.parse_args()

    main(args.output_directory, args.image_path)

if __name__ == '__main__':
    main()