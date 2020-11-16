# import numpy as np
# import matplotlib.pyplot as plt
# from gradcam import GradCam
from misc_functions import (
                            convert_to_grayscale,
                            preprocess_image,
                            # get_example_params,
                            # save_image,
                            # save_gradient_images
                            )

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt

from opacus import PrivacyEngine
from PIL import Image, ImageFilter

# function used to train a model 
# net           - trained neural net to test
# trainset      - set of training images
# trainloader   - dataloader loaded with images images in trainset
# device        - device to use to forward test image through network
# dp            - are we training with differential privacy?
def train_model(net,trainloader,trainset,device,dp):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=.001, momentum=.9)
    if dp == True:
        print('adding privacy engine')
        # if we are training with differential privacy, create the engine
        privacy_engine = PrivacyEngine(
            net,
            4,
            len(trainloader),
            alphas=[1, 10, 100],
            noise_multiplier=1.3,
            max_grad_norm=1.0,
        )   
        privacy_engine.attach(optimizer)

    for epoch in range(5):  # currently training for 5 epochs
        print(f'epoch: {epoch}')
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0


# Test a trained network. Takes a network, test loader, device, and classnames 
# and attempts to perform a classification
# net       - trained neural net to test
# testloader- dataloader with test images to sample from
# device    - device to use to forward test image through network
# classes   - names of classes labeling the predicted class 
# def test(net,testloader,device,classes):
#     dataiter = iter(testloader)
#     data = dataiter.next()
#     images, labels = data[0].to(device), data[1].to(device)
#     outputs = net(images).to(device)
#     # print images
#     # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#     plt.imshow(images[0].reshape((256,256,3)))
#     plt.show()

# Hacked this together to be used by the visualization modules to get a trained model
# that corresponds to the input parameters
# train     - should we train this model or not (if trained model does not exist, will be trained)
# dataset   - name of dataset currently only supporting 'cifar' ad 'mnist' 
# modeltype - type of neural network to be trained 'resnet', 'alex', 'vgg' - only have alexnets trained 
# dp        - are we training with differential privacy?
def get_trained_model(train,dataset,modeltype,dp):
    classes, trainloader, testloader, trainset, testset = get_test_train_loaders(dataset)
    device = get_device(train)

    if modeltype == 'alex':
        net = torchvision.models.alexnet(pretrained=False,progress=False).to(device)
    elif modeltype == 'resnet':
        net = torchvision.models.resnet18(pretrained=False,progress=True).to(device)
    else:
        net = torchvision.models.vgg16(pretrained=False,progress=False).to(device)

    if dp == True:
        PATH = './trained_models/' + dataset + '_' + modeltype + '_dp_net.pth'
        pass
    else:
        PATH = './trained_models/' + dataset + '_' + modeltype + 'net.pth'
    print(PATH)    
    if os.path.exists(PATH) and train == False:
        print('trained model exists, loading model')
        net.load_state_dict(torch.load(PATH))
        net = net.to(device)
    else:
        print('trained model does not exist, training model')
        train_model(net,trainloader,trainset,device,dp)
        torch.save(net.state_dict(), PATH)
        net = net.to(device)
    
    return net.to(device)

def get_test_train_loaders(dataset):
    transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    if dataset == "mnist":
        print(f"using:mnist")
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine') 
    else: # ie == cifar
        print(f"using other:{dataset}")
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    return classes, trainloader, testloader, trainset, testset

def get_classes(dataset):
    if dataset == "mnist":
        classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine') 
    else: # ie == cifar
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 
    return classes

def get_device(train):
    use_cuda = torch.cuda.is_available() and train
    # use_cuda = torch.cuda.is_available() 
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def get_example_class_imgpath(dataset,example_index):
    if dataset == 'mnist':
        example_list = (('./input_images/mnist_zero.png', 0),
                        ('./input_images/mnist_two.png', 2),
                        ('./input_images/mnist_eight.png', 8))
    else:
        example_list = (('./input_images/cifar_plane.png', 0),
                        ('./input_images/cifar_cat.png', 3),
                        ('./input_images/cifar_deer.png', 4))

    return example_list[example_index][0] 

def get_example_class_target(dataset,example_index):
    if dataset == 'mnist':
        example_list = (('./input_images/mnist_zero.png', 0),
                        ('./input_images/mnist_two.png', 2),
                        ('./input_images/mnist_eight.png', 8))
    else:
        example_list = (('./input_images/cifar_plane.png', 0),
                        ('./input_images/cifar_cat.png', 3),
                        ('./input_images/cifar_deer.png', 4))

    return example_list[example_index][1]

def get_example_input_image(dataset,example_index):
    img_path = get_example_class_imgpath(dataset, example_index)
    original_image = Image.open(img_path).convert('RGB')
    return original_image

def get_example_preprocessed_image(dataset,example_index):
    img_path = get_example_class_imgpath(dataset, example_index)
    original_image = Image.open(img_path).convert('RGB')
    return preprocess_image(original_image)

def get_example_filename(dataset,example_index):
    img_path = get_example_class_imgpath(dataset, example_index)
    return img_path[img_path.rfind('/')+1:img_path.rfind('.')]
