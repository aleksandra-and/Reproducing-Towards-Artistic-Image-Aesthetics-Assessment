from torchvision.transforms import transforms
from models.model import SAAN
from common import *
import argparse
import cv2 as cv

# transform
def transform(image):
    return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(image)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_path', type=str,
                        default='checkpoint/BAID/model_best.pth')
    parser.add_argument('--image_path', type=str,
                        default='immies/amazing_drawing.png')

    return parser.parse_args()

def inference(args):
    device = args.device
    checkpoint_path = args.checkpoint_path
    model = SAAN(num_classes=1)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'))
    model.eval()
    image_path = args.image_path
    with torch.no_grad():
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image_tensor = transform(image)\
            .to(device)\
            .unsqueeze(0)

        output = model(image_tensor)
        grade = output.item() * 10
        prefix = ""
        if grade <= 3:
            prefix = "Ew ew ew, that's at most a"
        elif grade <= 6:
            prefix = "Nyeh, I'll give that a"
        elif grade <= 9:
            prefix = "That's a solid"
        else:
            prefix = "Wow, that deserves a"
        print(prefix, grade)




if __name__ == '__main__':
    args = parse_args()
    inference(args)




