import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default=None, type=str)
args = parser.parse_args()

class_dict = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def main():
    device = torch.device("cuda")
    state = torch.load("checkpoints/02/checkpoint_150.pth")
    model = models.swin_v2_t()
    model.head = nn.Linear(768, 6)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(260, Image.Resampling.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(args.image_path)
    input_image = transform(image)
    input_image = input_image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.softmax(model(input_image), 1)
        list_value, list_index = torch.sort(output, 1, True)
        for i in range(6):
            print(f"{class_dict[list_index[0][i]]}[{list_value[0][i]:.4f}]")


if __name__ == "__main__":
    main()
