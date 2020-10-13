# Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch
## Introduction 
Dans ce tuto, vous apprendrez et utiliserez la technique d'apprentissage par transfert pour créer un models qui sera capable de predire si un homme porte son cache nez ou non.

resultat final

<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test%20result1.png" alt="output"/>

### Notion couvert par ce tutoriel 

1. [transform](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)

La transformation des données est le processus dans lequel vous extrayez les données de leur état source brut, cloisonné et normalisé et les transformez en données jointes, modélisées dimensionnellement, dénormalisées et prêtes pour l'analyse.

5. [models](https://pytorch.org/docs/stable/torchvision/models.html) 
The models subpackage contains definitions of models for addressing different tasks, including: image classification, pixelwise semantic segmentation, object detection, instance segmentation, person keypoint detection and video classification.

2. [datasets](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

3. [DataLoader](https://pytorch.org/docs/stable/torchvision/datasets.html)

3. [nn](https://pytorch.org/docs/stable/nn.html)

4. [optim](https://pytorch.org/docs/stable/optim.html#:~:text=optim-,torch.,easily%20integrated%20in%20the%20future.):torch.optim is a package implementing various optimization algorithms.

      
### prerequis
- Un ordinateur avec GPU
- une bonne maitrise du langage python.
- avoir les base sur l apprentisage profond  (neural network, convolutional neural network(CNN), etc. ) 

### insatlle l environement de travail  :
- Ordinateur local: vous pouvez suivre les instruction [ci](https://pytorch.org/get-started/locally/) pour pouvoir utilise pytorch dans votre ordinateur. 

- plateforme en tant que service: Kaggle Kernels est une plateforme gratuite permettant d'exécuter des notebooks Jupyter dans le navigateur. kaggle offre le  GPU gratuitement pour pouvoir facilement entraine le model.
vous pouvez vous enregistrez sur kaggle  [ici](https://www.kaggle.com/)

## Construire l application etape par etape  

### etape 0: Importe les daonne
Assurez-vous d'avoir téléchargé l'ensemble de données requis.

telechargez les [donne](https://www.kaggle.com/achilep/covid19-face-mask-data/download), pour tester nous allons utilise cet donne [ci](https://www.kaggle.com/achilep/covid19-face-mask-recognition-test-data).
### Step 1: Specify Data Loaders for the covid19-face-mask-data dataset
 
- ```transforms.Compose``` clubs juste toutes les transformations qui lui sont apportées. 
Donc, toutes les transformations dans le ```transforms.Compose``` sont appliqués à l'entrée un par un.


- ```transforms.RandomResizedCrop(224)```: Cela extraira un patch de taille (224, 224) de votre image d'entrée au hasard. Ainsi, il peut choisir ce chemin de haut en bas, en bas à droite ou n'importe où entre les deux. Donc, vous faites une augmentation des données dans cette partie. De plus, changer cette valeur ne fonctionnera pas bien avec les couches entièrement connectées de votre modèle, il n'est donc pas conseillé de changer cela.

- ```transforms.RandomHorizontalFlip()```: Une fois que nous avons notre image de taille (224, 224), nous pouvons choisir de la retourner. C'est une autre partie de l'augmentation des données.

- ```transforms.ToTensor()```: Cela convertit simplement votre image d'entrée en tenseur PyTorch.

- ```transforms.Resize(256)```: Tout d'abord, votre image d'entrée est redimensionnée pour être de taille (256, 256).

- ```transforms.CentreCrop(224)```: Recadre la partie centrale de l'image de la forme (224, 224).


- ```transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])```: Il ne s'agit que de la mise à l'échelle des données d'entrée et ces valeurs (moyenne et std) doivent avoir été précalculées pour votre ensemble de données. Il est également déconseillé de modifier ces valeurs.
                                                                            Dataloader est capable de cracher des échantillons aléatoires de nos données, de sorte que notre modèle n'aura pas à traiter l'ensemble de données à chaque fois. Cela rend la formation plus efficace.
                                                                            Nous spécifions le nombre d'images que nous voulons à la fois comme notre batch_size (donc 32 signifie que nous voulons obtenir 32 images à la fois). Nous voulons également mélanger nos images afin qu'elles soient entrées au hasard dans notre modèle d'IA.

- la ``datasets.ImageFolder()`` commande s'attend à ce que nos données soient organisées de la manière suivante: root / label / picture.png. En d'autres termes, les images doivent être triées dans des dossiers. Par exemple, toutes les images d'abeilles devraient être dans un dossier, toutes les images de fourmis devraient être dans un autre etc.

*La cellule de code ci-dessous écrit trois chargeurs de données distincts pour les ensembles de données d'entraînement, de validation et de test d'images humaines(situe a covid19-face-mask-data/face-mask-dataset/train, covid19-face-mask-data/face-mask-dataset/valid, and covid19-face-mask-data/face-mask-dataset/test, respectivement). 
Vous pouvez trouver ceci [documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) 
être une ressource utile. Si vous souhaitez augmenter vos données de formation et / ou de validation, découvrez la grande variété de [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!*

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
### etape 2: Define l Architecture du model
Utilisez l'apprentissage par transfert pour créer un CNN pour classer le masque facial. Utilisez le code ci-dessous et enregistrez votre modèle initialisé en tant que variable model_transfer.

L'apprentissage par transfert est une méthode d'apprentissage automatique dans laquelle un modèle développé pour une tâche est réutilisé comme point de départ d'un modèle sur une deuxième tâche.

C'est une approche populaire dans l'apprentissage en profondeur où des modèles pré-entraînés sont utilisés comme point de départ pour les tâches de vision par ordinateur et de traitement du langage naturel étant donné les vastes ressources de calcul et de temps nécessaires pour développer des modèles de réseaux neuronaux sur ces problèmes et des énormes sauts de compétences. qu'ils fournissent sur des problèmes connexes.

Il y a plusieurs [pretrain model](https://pytorch.org/docs/stable/torchvision/models.html) en pytorch : 

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



Nous utilisons le modèle Vgg16 dans ce tutoriel.

VGG16 est un modèle de réseau de neurones convolutif proposé par K. Simonyan et A. Zisserman de l'Université d'Oxford dans le document «Very Deep Convolutional Networks for Large-Scale Image Recognition». Le modèle atteint une précision de test de 92,7% dans le top 5 dans ImageNet, qui est un ensemble de données de plus de 14 millions d'images appartenant à 1000 classes.

<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/vgg16-neural-network-850x501.jpg" alt="Load the Model"/>

Voici une présentation plus intuitive du modèle VGG-16.
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/VGG-2-850x208.png" alt="Load the Model"/>

Voici les couches du modèle:

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

### Step 3: Specifies le  Loss Function et l Optimizer
Fonction d'erreur et de perte: dans la plupart des réseaux d'apprentissage, l'erreur est calculée comme la différence entre la sortie réelle et la sortie prévue.
La fonction utilisée pour calculer cette erreur est appelée fonction de perte.

#### Loss function
Les fonctions de perte sont des algorithmes mathématiques qui aident à mesurer à quel point un réseau neuronal apprend à obtenir le résultat réel. Dans l'apprentissage automatique, une fonction de perte est un algorithme mathématique qui évalue les performances d'un algorithme ML par rapport à son résultat souhaité. Il existe différentes fonctions de perte pour différents problèmes. Vous savez que le problème d'apprentissage automatique peut (en termes simples) être soit un problème de classification, soit un problème de régression. Cela implique que nous avons des fonctions de perte optimisées pour la classification et d'autres pour la régression. Pour n'en citer que quelques-uns, nous avons les fonctions de perte suivantes basées sur la classification (entropie croisée binaire, entropie croisée catégorique, similitude cosinus et autres). Nous avons également, pour n'en citer que quelques-uns, l'erreur quadratique moyenne (MSE), l'erreur en pourcentage absolue moyenne (MAPE), l'erreur absolue moyenne (MAE), pour n'en citer que quelques-uns, utilisées pour les problèmes fondés sur la régression.

#### An optimizer
Dans des phrases simples, un optimiseur peut essentiellement être appelé un algorithme qui aide un autre algorithme à atteindre ses performances maximales sans délai. En ce qui concerne l'apprentissage automatique (réseau de neurones), nous pouvons dire qu'un optimiseur est un algorithme mathématique qui aide notre fonction de perte à atteindre son point de convergence avec un délai minimum (et surtout, à réduire la possibilité d'explosion de gradient). Les exemples incluent, adam, descente de gradient stochastique (SGD), adadelta, rmsprop, adamax, adagrad, nadam etc.
Utilisez la cellule de code suivante pour spécifier une perte [function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html). Enregistrez la fonction de perte choisie sous critère_transfer et l'optimiseur sous optimizer_transfer ci-dessous.

```
import torch.optim as optim
criterion_transfer = nn.CrossEntropyLoss()

optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001)
```
### Step 4: Train and Validate the Model
Entraînez et validez votre modèle dans la cellule de code ci-dessous. [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) au chemin du fichier ```'model_transfer.pt'```.
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

### etape 5: Teste le Model
Essayez votre modèle sur l'ensemble de données de test. Utilisez la cellule de code ci-dessous pour calculer et imprimer la perte et la précision du test. Assurez-vous que la précision de votre test est supérieure à 60%.

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

### etape 6: Predire si la personne porte un cache nez ou non avec le  Model
Ecrire une fonction qui prend un chemin d'image en entrée et renvoie le masque si l'homme présent sur l'image porte un masque facial ou non en se basant sur la prédiction du modèle.
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
 ### etape 7: ecrire ton Algorithm 
 
Écrivez le run_app qu'une image d'un humain une impression ```This person is responsible, he wears his face mask!!!!``` quand une cette personne porte un visage 
 et ecrit ``` This person is irresponsible, he does not wear his face mask!!!!!``` quand un qui n'a pas de masque facial.
 

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
### etape 8: teste ta  fonction run_app 
Nous pouvons maintenant utiliser le test donne pour tester l'ensemble de données pour tester notre système.

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

### etape 10: integre opencv a notre project
Écrivez la méthode run_app_with_opencv qu'une image d'un humain est imprimée ```This person is responsible, he wears his face mask!!!!``` quand une cette personne porte un visage
et imprimer ``` This person is irresponsible, he does not wear his face mask!!!!!``` quand un qui n'a pas de masque facial. et en plus localisé le point culminant du visage d'une personne.
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
 ### etape 10: Teste ton Algorithm
 vous pouvez utiliser une image pour tester ```run_app_with_opencv```
 ```
## Execute your algorithm from Step 6 
## on 1 images on your computer.
## Feel free to use as many you want.
for file in np.array(glob("../input/covid19-face-mask-recognition-test-data/Covid19-face-mask-recognition-test-data/4.jpeg")):
    run_app_with_opencv(file)
  ```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test result with opencv.png" alt="result of the predition"/>

## conclusion
nous avons appris à utiliser le modèle pré-train pour accélérer la formation de notre modèle. 

## future work 
le projet peut être utilisé dans la fonction publique pour contrôler les personnes qui entrent, s'assurer qu'elles ont un masque facial.
1. nous pouvons intégrer notre modèle avec une webcam ou une caméra vidéo en utilisant opencv.
2. nous pouvons intégrer un système de notification.
2. nous pouvons intégrer notre modèle dans l'automatisation de porte ouverte, de telle manière que la porte ne s'ouvre que lorsqu'une personne porte un masque facial.
3. nous pouvons l'utiliser à l'école pour nous assurer que l'élève porte toujours son masque facial. 
