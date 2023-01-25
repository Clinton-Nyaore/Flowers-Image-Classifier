import argparse 
import time
import torch 
import numpy as np
import json
import sys
from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image


def parse():
    parser = argparse.ArgumentParser(description='Lets use our saved model to classify an image!')
    parser.add_argument('--image_input', default='ImageClassifier/flowers/train/1/image_06734.jpg', help='image file to classifiy')
    parser.add_argument('--model_checkpoint', default='flowers_check.pth', help='model used for classification')
    parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', default='ImageClassifier/cat_to_name.json', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    global in_args
    in_args = parser.parse_args()
    return in_args


def load_model():
    checkpoint = torch.load(in_args.model_checkpoint)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    resize = transforms.Resize(size=(256, 256))
    image = resize(image)
    crop = transforms.CenterCrop(224)
    image = crop(image)
    image = np.array(image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = image.transpose((2, 1, 0))
    
    return image


def classify_image(image_path, topk=5):
    topk = int(topk)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model = load_model()
        if (in_args.gpu):
           image = image.cuda()
           model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        predictions = model(image)
        probs, classes = torch.exp(predictions).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        results = zip(probs,classes)
        return results


def read_categories():
    if (in_args.category_names is not None):
        cat_file = in_args.category_names 
        jsfile = json.loads(open(cat_file).read())
        return jsfile
    return None


def display_prediction(results):
    cat_file = read_categories()
    idx = 0
    for percentage, classs in results:
        idx = idx + 1
        percentage = str(round(percentage, 4) * 100.) + '%'
        if (cat_file):
            classs = cat_file.get(str(classs),'None')
        else:
            classs = ' Class {}'.format(str(classs))
        print("{}.{} is {}".format(idx, classs, percentage))
    return None


def main():
    in_args = parse() 
    if (in_args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled -- but no GPU detected")
    if (in_args.top_k is None):
        top_k = 5
    else:
        top_k = in_args.top_k
    image_path = in_args.image_input
    prediction = classify_image(image_path, top_k)
    display_prediction(prediction)
    return prediction

main()