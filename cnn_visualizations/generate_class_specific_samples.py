"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import SGD
from torchvision import models
import torchvision
import train_models

import matplotlib.pyplot as plt
from misc_functions import preprocess_image, recreate_image, save_image, convert_to_grayscale


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        # Create the folder to export images if not exists
        # if not os.path.exists('../generated/class_'+str(self.target_class)):
        #     os.makedirs('../generated/class_'+str(self.target_class))

    def generate(self, iterations=150):
        """Generates class specific image

        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {150})

        Returns:
            np.ndarray -- Final maximally activated class image
        """
        initial_learning_rate = 6
        for i in range(1, iterations):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)

            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]

            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)

        return self.created_image
        # return self.processed_image


def class_act_example(train,dataset,modeltype,dp):
    f, axarr = plt.subplots(3,2) 
    f.set_figheight(10)
    f.set_figwidth(10)


    pretrained_model = train_models.get_trained_model(train,dataset,modeltype,dp)
    classes = train_models.get_classes(dataset)


    example_index = 0
    target_class = train_models.get_example_class_target(dataset,example_index)
    original_image = train_models.get_example_input_image(dataset,example_index)
    class_grads = ClassSpecificImageGeneration(pretrained_model, target_class).generate()
    axarr[example_index,0].imshow(original_image)
    axarr[example_index,0].title.set_text(f'Original Image - {classes[target_class]}')
    axarr[example_index,1].imshow(class_grads)
    axarr[example_index,1].title.set_text(f'Gradient Visualization Class Activation - {classes[target_class]}')
    
    example_index = 1
    target_class = train_models.get_example_class_target(dataset,example_index)
    original_image = train_models.get_example_input_image(dataset,example_index)
    class_grads = ClassSpecificImageGeneration(pretrained_model, target_class).generate()

    axarr[example_index,0].imshow(original_image)
    axarr[example_index,0].title.set_text(f'Original Image - {classes[target_class]}')
    axarr[example_index,1].imshow(class_grads)
    axarr[example_index,1].title.set_text(f'Gradient Visualization Class Activation - {classes[target_class]}')

    example_index = 2
    target_class = train_models.get_example_class_target(dataset,example_index)
    original_image = train_models.get_example_input_image(dataset,example_index)
    class_grads = ClassSpecificImageGeneration(pretrained_model, target_class).generate()

    axarr[example_index,0].imshow(original_image)
    axarr[example_index,0].title.set_text(f'Original Image - {classes[target_class]}')
    axarr[example_index,1].imshow(class_grads)
    axarr[example_index,1].title.set_text(f'Gradient Visualization Class Activation - {classes[target_class]}')
    
    # plt.show()    

    # print('Class Specific complete')


if __name__ == '__main__':
    train       = False 
    dataset     = 'cifar'
    modeltype   = 'alex'
    dp          = False
    class_act_example(train,dataset,modeltype,dp)