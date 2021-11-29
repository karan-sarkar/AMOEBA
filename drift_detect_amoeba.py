from torchvision import transforms
from others.util import *
from PIL import Image
import tqdm
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
args = parser.parse_args()

# Load model checkpoint
checkpoint = args.ckpt
model,_ = torch.load(checkpoint)
model = model.to(device)
if not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
filter_obj = 'truck'
filter_limit = 1
query_obj = 'car'
counts = []

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores,predicted_locs2,predicted_scores2 = model(image.unsqueeze(0))
    return float((predicted_locs - predicted_locs2).abs().mean()), float((predicted_scores - predicted_scores2).abs().mean())
    


if __name__ == '__main__':
    df = pd.read_csv('drift_data.csv')
    print(df)
    
    img_list = list(df['Name'])
    df[checkpoint + 'loc'] = 0
    df[checkpoint + 'cls'] = 0
  
    for i, img_path in tqdm.tqdm(enumerate(img_list)):
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        locs, cls = detect(original_image, min_score=0.2, max_overlap=0.45, top_k=200)
        df.loc[i, checkpoint + 'loc'] = locs
        df.loc[i, checkpoint + 'cls'] = cls
        
    df.to_csv('drift_data.csv', index=False)
        
