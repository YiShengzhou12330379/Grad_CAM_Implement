import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

from model import Net

#Class for extracting activations and back-propagation gradients
class FeatureExtractor():
    
    def __init__(self, model, target_module, target_index, target_layers):
        self.model = model
        self.target_module = target_module
        self.target_index = target_index
        self.target_layer = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        
        for top_index, top_module in self.model._modules.items():
            if len(top_module._modules) > 0:
                for bottleneck_index, bottleneck_module in top_module._modules.items():
                    if bottleneck_index == '0':
                        out = x
                        residual = x
                        for layer_name, module in bottleneck_module._modules.items():
                            if layer_name == 'downsample':
                                residual = module(out)
                                x += residual
                            else:
                                print(module)
                                x = module(x)  # Forward
                            
                            if (top_index == self.target_module and bottleneck_index == self.target_index) and layer_name == self.target_layer:
                                outputs += [x]
                    else:
                        residual = x
                        for layer_name, module in bottleneck_module._modules.items():
                            print(module)
                            x = module(x)  # Forward
                            if layer_name == 'bn3:
                                x += residual
                        
                            if (top_index == self.target_module and bottleneck_index == self.target_index) and layer_name == self.target_layer:
                                outputs += [x]
            else:
                print(top_module)
                x = top_module(x)
        return outputs, x

#Forward propagation
class ModelOutputs():
    def __init__(self, model, target_module, target_index, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_module, target_index, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = self.model.gap(output)
        output = torch.squeeze(output)
        output = self.model.classifier(output)
        return target_activations, output

#preprocess image
def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

#write heatmap to image file
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite('src_gradcam/cam.jpg', np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_module_name, target_index_name, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_module_name, target_index_name, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        #zero grads
        for top_index, top_module in self.model._modules.items():
            if len(top_module._modules) > 0:
                for bottleneck_index, bottleneck_module in top_module._modules.items():
                    for layer_name, module in bottleneck_module._modules.items():
                        if layer_name == 'downsample':
                            for sample_name, sample_module in module._modules.items():
                                sample_module.zero_grad()
                        else:
                            module.zero_grad()
            else:
                top_module.zero_grad()
    
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

#Guide back-propagation
class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='data/image/test/000001.jpg',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    args = get_args()
    
    #loading model
    model = Net()
    PATH = 'saved_model/' + 'resnet_finetuning.tar'
    model.load_state_dict(torch.load(PATH))

    grad_cam = GradCam(
        model = model,
        target_module_name = '7',
        target_index_name = '2',
        target_layer_names = 'conv3',
        use_cuda=args.use_cuda)

    img = cv2.imread('data/image/test/' + args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    #feedback bad area
    target_index = 0

    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model = model, use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    utils.save_image(torch.from_numpy(gb), 'src_gradcam/gb.jpg')

    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), 'src_gradcam/cam_gb.jpg')
