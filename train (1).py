import argparse
import os
import torch
import time
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', default='ImageClassifier/flowers', help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='models to use i.e vgg or densenet')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_size', help='number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu', action='store_true', help='gpu')
    global in_args
    in_args = parser.parse_args()
    return in_args

def validation():
    print('Validating parameters ...')
    if (in_args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled -- but no GPU detected")
    if (not os.path.isdir(in_args.data_directory)):
        raise Exception('The directory does not exist!')
    data_dir = os.listdir(in_args.data_directory)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('Missing: test, train or valid sub-directories')
    if in_args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose one of: vgg or densenet')   
    print(print('Done validating the parameters'))

def process_data(data_dir):
    print('Processing data into the loaders ... ')
    train_dir, test_dir, valid_dir = data_dir 
    
    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    val_test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) 
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64) 
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64) 
    
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    loaders = {'train':train_dataloader,'valid':val_dataloader,'test':test_dataloader,'labels':cat_to_name}
    print('Done processing data into the loaders')
    return loaders


def get_data():
    print('Retreiving data ...')
    train_dir = in_args.data_directory + '/train'
    test_dir = in_args.data_directory + '/test'
    valid_dir = in_args.data_directory + '/valid'
    data_dir = [train_dir, test_dir, valid_dir]
    print('Done retreiving data')
    return process_data(data_dir)


def build_model(data):
    print('Building our model ... ')
    if (in_args.arch is None):
        arch_type = 'vgg'
    else:
        arch_type = in_args.arch
    if (arch_type == 'vgg'):
        model = models.vgg11(pretrained=True)
        input_size= 25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_size = 1024
    if (in_args.hidden_size is None):
        hidden_size = 1024
    else:
        hidden_size = in_args.hidden_size
    for param in model.parameters():
        param.requires_grad = False
    hidden_size = int(hidden_size)
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(input_size, hidden_size)),
                                ('relu', nn.ReLU()),
                                ('dropout1', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(hidden_size, 512)),
                                ('relu2', nn.ReLU()),
                                ('dropout2', nn.Dropout(p=0.2)),
                                ('fc3', nn.Linear(512, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                        ]))
    model.classifier = classifier
    print('Done building our model')
    return model

def test_accuracy(model, loader, device='cpu'):   
    criterion = nn.NLLLoss()
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            log_preds = model(images)
            test_loss += criterion(log_preds, labels)

            log_preds = torch.exp(log_preds)
            top_p, top_class = log_preds.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
    return accuracy 


def train(model, data):
    print('Training our model ... ')

    if (in_args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = in_args.learning_rate
    if (in_args.epochs is None):
        epochs = 1
    else:
        epochs = in_args.epochs
    if (in_args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
    
    learn_rate = float(learn_rate)
    epochs = int(epochs)
    
    train_dataloader = data['train']
    val_dataloader = data['valid']
    test_dataloader = data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    steps = 0
    train_losses, test_losses = [], []

    if torch.cuda.is_available():
        model.cuda()

    for e in range(epochs):
        running_loss = 0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            model = model.to(device)
            optimizer.zero_grad()
            log_preds = model(images)
            loss = criterion(log_preds, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:

            test_loss = 0
            accuracy = 0
        
            with torch.no_grad():

                for images, labels in val_dataloader:
                    images, labels = images.to(device), labels.to(device)
            
                    log_preds = model(images)
                    test_loss += criterion(log_preds, labels)

                    log_preds = torch.exp(log_preds)
                    top_p, top_class = log_preds.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss/len(train_dataloader))
            test_losses.append(test_loss/len(val_dataloader))

            print("Epoch {}/{} ... ".format(e+1, epochs),
                  "Training Loss : {:.3f} ... ".format(running_loss/len(train_dataloader)),
                  "Test Loss : {:.3f} ... ".format(test_loss/len(val_dataloader)),
                  "Test Accuracy : {:.3f}".format(accuracy/len(val_dataloader)))

    print('Done training our model')
    test_result = test_accuracy(model, test_dataloader, device)
    print('Final accuracy on test set: {}'.format(test_result))
    return model


def save_model(model):
    print('Saving our model ... ')
    if (in_args.save_dir is None):
        save_dir = 'flowers_check.pth'
    else:
        save_dir = in_args.save_dir
    checkpoint = {
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    print('Done saving our model')
    return 0


def create_model():
    validation()
    data = get_data()
    model = build_model(data)
    model = train(model,data)
    save_model(model)
    return None


def main():
    print("Creating the model started ... ")
    in_args = parse()
    create_model()
    print("Model creation finished successfully !!!")
    return None

main()