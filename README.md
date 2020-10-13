# Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch
## Introduction 
Dans ce tuto, vous apprendrez et utiliserez la technique d'apprentissage par transfert pour créer un models qui sera capable de predire si un homme porte son cache nez ou non.

resultat final

<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/tree/main/Resource/readme_image/test result1.png" alt="output"/>

### Notion couvert par ce tutoriel 

1. [transform](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)

La transformation des données est le processus dans lequel vous extrayez les données de leur état source brut, cloisonné et normalisé et les transformez en données jointes, modélisées dimensionnellement, dénormalisées et prêtes pour l'analyse.

5. [models](https://pytorch.org/docs/stable/torchvision/models.html) 
The models subpackage contains definitions of models for addressing different tasks, including: image classification, pixelwise semantic segmentation, object detection, instance segmentation, person keypoint detection and video classification.

2. [datasets](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

3. [DataLoader](https://pytorch.org/docs/stable/torchvision/datasets.html)

3. [nn](https://pytorch.org/docs/stable/nn.html)

4. [optim](https://pytorch.org/docs/stable/optim.html#:~:text=optim-,torch.,easily%20integrated%20in%20the%20future.):torch.optim is a package implementing various optimization algorithms.

      
### requirement 
- Computer with GPU
- Good knowledge of python.
- Basic knowledge of deep learning (neural network, convolutional neural network(CNN), etc. ) 

### Seting up the working environment :
- Local computer: you can follow the instruction [here](https://pytorch.org/get-started/locally/) to set up pytorch in computer. 

- platform as as service: Kaggle Kernels is a free platform to run Jupyter notebooks in the browser. kaggle provide free GPU to train you model.
you can Sign in [here](https://www.kaggle.com/)

## Building the App step by step  

### Step 0: Import Datasets
Make sure that you've downloaded the required dataset.

Download the [dataset](https://www.kaggle.com/achilep/covid19-face-mask-data/download), For testing we are using this [dataset](https://www.kaggle.com/achilep/covid19-face-mask-recognition-test-data).
### Step 1: Specify Data Loaders for the covid19-face-mask-data dataset
 
- ```transforms.Compose``` just clubs all the transforms provided to it. So, all the transforms in the ```transforms.Compose``` are applied to the input one by one.


- ```transforms.RandomResizedCrop(224)```: This will extract a patch of size (224, 224) from your input image randomly. So, it might pick this path from topleft, bottomright or anywhere in between. So, you are doing data augmentation in this part. Also, changing this value won't play nice with the fully-connected layers in your model, so not advised to change this.

- ```transforms.RandomHorizontalFlip()```: Once we have our image of size (224, 224), we can choose to flip it. This is another part of data augmentation.

- ```transforms.ToTensor()```: This just converts your input image to PyTorch tensor.

- ```transforms.Resize(256)```: First your input image is resized to be of size (256, 256)

- ```transforms.CentreCrop(224)```: Crops the center part of the image of shape (224, 224)


- ```transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])```: This is just input data scaling and these values (mean and std) must have been precomputed for your dataset. Changing these values is also not advised.
Dataloader is able to spit out random samples of our data, so our model won’t have to deal with the entire dataset every time. This makes training more efficient.
We specify how many images we want at once as our batch_size (so 32 means we want to get 32 images at one time). We also want to shuffle our images so it gets inputted randomly into our AI model.


- The ``datasets.ImageFolder()`` command expects our data to be organized in the following way: root/label/picture.png. In other words, the images should be sorted into folders. For example, all the pictures of bees should be in one folder, all the pictures of ants should be in another etc.

*The code cell below write three separate data loaders for the training, validation, and test datasets of humans images (located at covid19-face-mask-data/face-mask-dataset/train, covid19-face-mask-data/face-mask-dataset/valid, and covid19-face-mask-data/face-mask-dataset/test, respectively). You may find this [documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource. If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!*

```
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
from torchvision import datasets

datadir = {

    'train': '../input/covid19-face-mask-data/face-mask-dataset/train/',
    'valid': '../input/covid19-face-mask-data/face-mask-dataset/validation/',
    'test': '../input/covid19-face-mask-data/face-mask-dataset/test'
}

trns_normalize = transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])

transform_transfer = {}
transform_transfer['train'] = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        trns_normalize
    ])
transform_transfer['valid'] = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        trns_normalize
    ])
transform_transfer['test'] = transform_transfer['valid']

# Trying out an idiom found in the pytorch docs
datafolder_transfer = {x: datasets.ImageFolder(datadir[x], transform=transform_transfer[x]) for x in ['train', 'valid', 'test']}

batch_size = 20
num_workers = 0

# Trying out an idiom found in the pytorch docs
loaders_transfer = {x: torch.utils.data.DataLoader(datafolder_transfer[x], batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True) 
for x in ['train', 'valid', 'test']}
```

```
loaders_scratch = {}
loaders_scratch['train'] = torch.utils.data.DataLoader(datafolder_transfer['train'], batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
loaders_scratch['valid'] = torch.utils.data.DataLoader(datafolder_transfer['valid'], batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
loaders_scratch['test'] = torch.utils.data.DataLoader(datafolder_transfer['test'], batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
```
### Step 2: Define the Model Architecture
Use transfer learning to create a CNN to classify the face mask . Use the code below, and save your initialized model as the variable model_transfer.

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.

there are various [pretrain model](https://pytorch.org/docs/stable/torchvision/models.html) in pytorch : 

- resnet18 
- alexnet 
- vgg16 
- squeezenet 
- densenet 
- inception 
- googlenet 
- shufflenet 
- mobilenet 
- resnext50_32x4d 
- wide_resnet50_2 
- mnasnet 


We are using Vgg16 model in this tutorial.

VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes.

<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/vgg16-neural-network-850x501.jpg" alt="Load the Model"/>

Here is a more intuitive layout of the VGG-16 Model.
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/VGG-2-850x208.png" alt="Load the Model"/>

The following are the layers of the model:

Convolutional Layers = 13
Pooling Layers = 5
Dense Layers = 3
Let us explore the layers in detail:

1. Input: Image of dimensions (224, 224, 3).
2. Convolution Layer Conv1:
    - Conv1-1: 64 filters
    - Conv1-2: 64 filters and Max Pooling
    - Image dimensions: (224, 224)
3. Convolution layer Conv2: Now, we increase the filters to 128
    - Input Image dimensions: (112,112)
    - Conv2-1: 128 filters
    - Conv2-2: 128 filters and Max Pooling
4. Convolution Layer Conv3: Again, double the filters to 256, and now add another convolution layer
    - Input Image dimensions: (56,56)
    - Conv3-1: 256 filters
    - Conv3-2: 256 filters
    - Conv3-3: 256 filters and Max Pooling
5. Convolution Layer Conv4: Similar to Conv3, but now with 512 filters
    - Input Image dimensions: (28, 28)
    - Conv4-1: 512 filters
    - Conv4-2: 512 filters
    - Conv4-3: 512 filters and Max Pooling
6. Convolution Layer Conv5: Same as Conv4
    - Input Image dimensions: (14, 14)
    - Conv5-1: 512 filters
    - Conv5-2: 512 filters
    - Conv5-3: 512 filters and Max Pooling
The output dimensions here are (7, 7). At this point, we flatten the output of this layer to generate a feature vector

7. Fully Connected/Dense FC1: 4096 nodes, generating a feature vector of size(1, 4096)
8. Fully ConnectedDense FC2: 4096 nodes generating a feature vector of size(1, 4096)
9. Fully Connected /Dense FC3: 4096 nodes, generating 1000 channels for 1000 classes. This is then passed on to a Softmax activation function
10. Output layer

As you can see, the model is sequential in nature and uses lots of filters. At each stage, small 3 * 3 filters are used to reduce the number of parameters all the hidden layers use the ReLU activation function. Even then, the number of parameters is 138 Billion – which makes it a slower and much larger model to train than others.

Additionally, there are variations of the VGG16 model, which are basically, improvements to it, like VGG19 (19 layers). You can find a detailed explanation
Let us now explore how to train a VGG-16 model on our dataset

```
import torchvision.models as models
import torch.nn as nn


##  Specify model architecture 
model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.features.parameters():
    param.requires_grad = False
    
### make changes to final fully collected layers
n_inputs = model_transfer.classifier[6].in_features
last_layer = nn.Linear(n_inputs, 133)
model_transfer.classifier[6] = last_layer
# check if CUDA is available
use_cuda = torch.cuda.is_available()

if use_cuda:
    model_transfer = model_transfer.cuda()
  ```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/pretrainmodel.png" alt="Load the Model"/>

```
print(model_transfer)
```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/print-modeltrasnfert.png" alt="Load the Model"/>

### Step 3: Specify Loss Function and Optimizer
Error and Loss Function: In most learning networks, error is calculated as the difference between the actual output and the predicted output.
The function that is used to compute this error is known as Loss Function.

#### Loss function
loss functions are mathematical algorithms that helps measure how close a neural net learns to getting the actual result. In machine learning, a loss function is a mathematical algorithm that evaluates the performance of an ML algorithm with respect to its desired result. There are various loss functions for various problems. You are aware that machine learning problem can (in basic terms) be either a classification problem or a regression problem. This implies that we do have optimized loss functions for classification and others for regression. To mention a few, we do have the following loss functions as classification based (binary cross entropy, categorical cross entropy, cosine similarity and others). We also have, mean squared error (MSE), mean absolute percentage error (MAPE), mean absolute error (MAE), just to mention a few, used for regression based problems.


#### An optimizer
In simple sentences, an optimizer can basically be referred to as an algorithm that helps another algorithm to reach its peak performance without delay. With respect to machine learning (neural network), we can say an optimizer is a mathematical algorithm that helps our loss function reach its convergence point with minimum delay (and most importantly, reduce the posibility of gradient explosion). Examples include, adam, stochastic gradient descent (SGD), adadelta, rmsprop, adamax, adagrad, nadam etc.

Use the next code cell to specify a loss [function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html). Save the chosen loss function as criterion_transfer, and the optimizer as optimizer_transfer below.

```
import torch.optim as optim
criterion_transfer = nn.CrossEntropyLoss()

optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001)
```
### Step 4: Train and Validate the Model
Train and validate your model in the code cell below. [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath ```'model_transfer.pt'```.
```
import numpy as np
from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        torch.enable_grad()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # train_loss += loss.item()*data.size(0) # From cifar10_cnn_solution.py
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        
        ######################
        # validate the model #
        ######################
        model.eval()
        torch.no_grad()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))
        
        ##  save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model
    ```
    
    ```
    # train the model
n_epochs =4 #25
model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/train.png" alt="Load the Model"/>

```
print(model_transfer)
```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/after_train.png" alt="Load the Model"/>

### Step 5: Test the Model
Try out your model on the test dataset . Use the code cell below to calculate and print the test loss and accuracy. Ensure that your test accuracy is greater than 60%.

```
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function
```

```
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test_accuraty.png" alt="Test Accuracy"/>

### Step 6: Predict if a human is wearing a face mask or not  with the Model
Write a function that takes an image path as input and returns the mask if the man present on the image is wearing a face mask or not base on the prediction of the model.
```
###  Write a function that takes a path to an image as input
### and returns the prediction of the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in datafolder_transfer['train'].classes]

def predict_transfer(img_path):
    # load the image and return the predicted result
    img = Image.open(img_path).convert('RGB')
    size = (224, 224) # ResNet image size requirements
    transform_chain = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])

    img = transform_chain(img).unsqueeze(0)

    if use_cuda:
        img = img.cuda()

    model_out = model_transfer(img)

    if use_cuda:
        model_out = model_out.cpu()
    
    prediction = torch.argmax(model_out)
    
    return class_names[prediction]  # predicted class label
  ```
 ### Step 7: Write your Algorithm 
 Write the run_app that an image of a human an print ```This person is responsible, he wears his face mask!!!!``` when a that person is wearing a face 
 and print ``` This person is irresponsible, he does not wear his face mask!!!!!``` when a that does not have a face mask.
 

```$xslt
import matplotlib.pyplot as plt 
def run_app(img_path):
   
    result = predict_transfer(img_path)
    #print(result )
    # display the image, along with bounding box
    if result == " mask" :
        print('This person is responsible, he wears his face mask!!!!!')
    else :
        print('This person is irresponsible, he does not wear his face mask!!!!!')
    img = plt.imread(img_path, 3)
    plt.imshow(img)
    plt.show()
```
### Step 8: test our function run_app 
We can now use how test dataset to test our system.

```$xslt
## Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as you want.

for file in np.array(glob("../input/covid19-face-mask-recognition-test-data/Covid19-face-mask-recognition-test-data/*")):
    run_app(file)
```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test result1.png" alt="result of the predition"/>
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test result1.png" alt="result of the predition"/>
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test result1.png" alt="result of the predition"/>

### Step 9: optional integrate opencv to the project
Write the run_app_with_opencv method that an image of a human an print ```This person is responsible, he wears his face mask!!!!``` when a that person is wearing a face 
 and print ``` This person is irresponsible, he does not wear his face mask!!!!!``` when a that does not have a face mask. and in addition located the highlight the face of a person .
 ```
import matplotlib.pyplot as plt 
import cv2                                       
%matplotlib inline 
def run_app_with_opencv(img_path):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('../input/xmldoc/haarcascade_frontalface_default.xml')

    # load color (BGR) image
    img = cv2.imread(img_path)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    # get bounding box for each detected face
    for (x,y,w,h) in faces:
        # add bounding box to color image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predict_transfer(img_path)
    # display the image, along with bounding box
    if result == " mask" :
        print("This person is responsible, he wears his face mask!!!!!" )
    else :
        print('This person is irresponsible, he does not wear his face mask!!!!!')

    plt.imshow(cv_rgb)
    plt.show()
 ```
 ### Step 10: Test Your Algorithm
 you can use one image to test ```run_app_with_opencv```
 ```
## Execute your algorithm from Step 6 
## on 1 images on your computer.
## Feel free to use as many you want.
for file in np.array(glob("../input/covid19-face-mask-recognition-test-data/Covid19-face-mask-recognition-test-data/4.jpeg")):
    run_app_with_opencv(file)
  ```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test result with opencv.png" alt="result of the predition"/>

## summary 
we have learn how to use pre- train model to speed up the training of our model. 

## future work 
the project can be use in the public service to control people that are entry, the make sure that they have they face mask.
1. we can integrate our model with a webcam or video camera using opencv.
2. we can integrate a notification system .
2. we can integrate our model in automation door open, in such a way that the door will open only when a person is wearing a face mask .
3. we can use it in school to make sure that the student allway wear they face mask. 
