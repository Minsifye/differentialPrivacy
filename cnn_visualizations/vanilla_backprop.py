"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

import train_models
import matplotlib.pyplot as plt
from misc_functions import get_example_params, convert_to_grayscale, save_gradient_images
import numpy as np


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def main():
    train       = False
    dataset     = 'cifar'
    modeltype   = 'alex'
    dp          = True
    vanilla_backprop_example(train,dataset,modeltype,dp)
    pass
def vanilla_backprop_example(train,dataset,modeltype,dp):
    f, axarr = plt.subplots(3,2) 
    f.set_figheight(10)
    f.set_figwidth(10)

    classes = train_models.get_classes(dataset)
    pretrained_model = train_models.get_trained_model(train,dataset,modeltype,dp)
    VBP = VanillaBackprop(pretrained_model)

    print('model loaded, forwarding examples')

    # show example index 0
    example_index = 0
    target_class = train_models.get_example_class_target(dataset,example_index)
    original_image = train_models.get_example_input_image(dataset,example_index)
    prep_img = train_models.get_example_preprocessed_image(dataset,example_index)
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    axarr[example_index,0].imshow(original_image)
    axarr[example_index,0].title.set_text(f'Original Image - {classes[target_class]}')
    axarr[example_index,1].imshow(grayscale_vanilla_grads.squeeze(0))
    axarr[example_index,1].title.set_text(f'Gradient Visualization Vanilla Backprop - {classes[target_class]}')
    
    example_index = 1
    target_class = train_models.get_example_class_target(dataset,example_index)
    original_image = train_models.get_example_input_image(dataset,example_index)
    prep_img = train_models.get_example_preprocessed_image(dataset,example_index)
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    axarr[example_index,0].imshow(original_image)
    axarr[example_index,0].title.set_text(f'Original Image - {classes[target_class]}')
    axarr[example_index,1].imshow(grayscale_vanilla_grads.squeeze(0))
    axarr[example_index,1].title.set_text(f'Gradient Visualization Vanilla Backprop - {classes[target_class]}')
    
    example_index = 2
    target_class = train_models.get_example_class_target(dataset,example_index)
    original_image = train_models.get_example_input_image(dataset,example_index)
    prep_img = train_models.get_example_preprocessed_image(dataset,example_index)
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    axarr[example_index,0].imshow(original_image)
    axarr[example_index,0].title.set_text(f'Original Image - {classes[target_class]}')
    axarr[example_index,1].imshow(grayscale_vanilla_grads.squeeze(0))
    axarr[example_index,1].title.set_text(f'Gradient Visualization Vanilla Backprop - {classes[target_class]}')

    plt.show()    


    print('Vanilla backprop completed')

if __name__ == '__main__':
    main()