from crackdetection.settings import BASE_DIR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import os
from PIL import Image
import time
import copy
import random
import cv2
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

#loading saved model
loaded_model = pickle.load(open('./models/trained_model.sav', 'rb'))

#computed mean and std deviation for dataset
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

## Define data augmentation and transforms
chosen_transforms = {'train': transforms.Compose([
        transforms.RandomResizedCrop(size=227),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
]), 'val': transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
]),
}

idx_to_class = {0:'Negative', 1:'Positive'}

t1 = ""
#prediction on test image
def predict(model, test_image, print_class = False):

    transform = chosen_transforms['val']

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 227, 227).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 227, 227)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        class_name = idx_to_class[topclass.cpu().numpy()[0][0]]
        if print_class:
            print("Output class :  ", class_name)
    return class_name

def predict_on_cracks(input_image, height=227, width=227, save_cracks = False):
    im = cv2.imread(input_image)
    t2 = input_image.split('/')
    #print(t2)
    t1 = t2[-1]
    imgheight, imgwidth, channels = im.shape
    k=0
    pcrack=0
    pno_crack=0
    output_image = np.zeros_like(im)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            a = im[i:i+height, j:j+width]
            ## discard image cracks that are not full size
            predicted_class = predict(loaded_model,Image.fromarray(a))
            ## save image
            file, ext = os.path.splitext(input_image)
            image_name = file.split('/')[-1]
            folder_name = 'out_' + image_name
            ## Put predicted class on the image
            if predicted_class == 'Positive':
                pcrack+=1
                color = (0,0, 255)
            else:
                pno_crack+=1
                color = (0, 255, 0)
            cv2.putText(a, predicted_class, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 0.7, color, 1, cv2.LINE_AA)
            b = np.zeros_like(a, dtype=np.uint8)
            b[:] = color
            add_img = cv2.addWeighted(a, 0.9, b, 0.1, 0)
            ## Save cracks
            if save_cracks:
                if not os.path.exists(os.path.join('real_images', folder_name)):
                    os.makedirs(os.path.join('real_images', folder_name))
                filename = os.path.join('real_images', folder_name,'img_{}.png'.format(k))
                cv2.imwrite(filename, add_img)
            output_image[i:i+height, j:j+width,:] = add_img
            k+=1

    pcrack/=k
    pno_crack/=k
    print("\nPercentage of crack: ",pcrack*100,"%")
    if pcrack>=0.7:
        print("WARNING!!")
        s = ("SPEED SHOULD BE < 30 KM/HR")
    elif pcrack>=0.4 and pcrack<0.7:
        print("WARNING!!")
        s = ("SPEED SHOULD BE < 50 KM/HR")
    elif pcrack<0.2:
        s = ("SPEED SHOULD BE < 60 KM/HR")
    else:
        print("WARNING!!")
        s = ("SPEED SHOULD BE < 70 KM/HR")
    ## Save output image
    #cv2.imwrite(os.path.join('real_images','predictions', folder_name+ '.jpg'), output_image)
    fp=os.path.join(BASE_DIR,'ouputimg')
    
    fp+='/'+t1
    cv2.imwrite(fp, output_image)
    print("nehal",fp)
    return fp,output_image, pcrack, s


# #testing an example
# plt.figure(figsize=(10,10))
# output_image = predict_on_cracks('images/real_images/concrete_crack3.jpeg', 128, 128)
# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
# plt.show()