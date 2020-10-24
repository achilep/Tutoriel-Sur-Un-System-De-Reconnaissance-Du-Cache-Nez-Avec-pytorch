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

4. [optim](https://pytorch.org/docs/stable/optim.html#:~:text=optim-,torch.,easily%20integrated%20in%20the%20future.)
torch.optim est un package implémentant divers algorithmes d'optimisation.

      
### prerequis
- Un ordinateur avec GPU
- Une bonne maitrise du langage python.
- Avoir les base sur l apprentisage profond  (neural network, convolutional neural network(CNN), etc. ) 

### Insatller l environement de travail  :
- Ordinateur local: vous pouvez suivre les instruction [ci](https://pytorch.org/get-started/locally/) pour pouvoir utilise pytorch dans votre ordinateur. 

- plateforme en tant que service: Kaggle Kernels est une plateforme gratuite permettant d'exécuter des notebooks Jupyter dans le navigateur. kaggle offre le  GPU gratuitement pour pouvoir facilement entraine le model.
vous pouvez vous enregistrez sur kaggle  [ici](https://www.kaggle.com/)

## Construction de l intelligence artificiel etape par etape  

### Etape0 acquisition des donnees nécessaire à l'apprentissage du model
Assurez-vous d'avoir téléchargé l'ensemble de données requis.

Telechargez les donnees nécessaire à l'apprentissage du model [ici]((https://www.kaggle.com/achilep/covid19-face-mask-data/download)) , pour tester nous allons utilise cet donne [ci](https://www.kaggle.com/achilep/covid19-face-mask-recognition-test-data).
### Etape1 Spécifier les chargeurs de données pour l'ensemble de données covid19-face-mask-data 
 
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
### Etape2 Define l Architecture du model
Dans cette section, vous apprendrez à utiliser des réseaux pré-formés pour résoudre des problèmes complexes de vision par ordinateur. Plus précisément, vous utiliserez les réseaux formés ImageNet [disponibles auprès de torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html).


ImageNet est un vaste ensemble de données avec plus d'un million d'images étiquetées dans 1000 catégories. Il est utilisé pour entraîner des réseaux de neurones profonds à l'aide d'une architecture appelée couches convolutives. Je n'entrerai pas dans les détails des réseaux convolutifs ici, mais si vous voulez en savoir plus sur eux,s'il te plait regarde [ça](https://www.youtube.com/watch?v=2-Ol7ZB0MmU).


Une fois formés, ces modèles fonctionnent à merveille et comportent des détecteurs d'images sur lesquelles ils n'ont pas été formés. L'utilisation d'un réseau pré-formé sur des images, et non dans l'ensemble d'apprentissage, est appelée apprentissage par transfert. Ici, nous utiliserons l'apprentissage par transfert pour former un réseau capable de classer notre visage avec un masque et des photos de visage sans masque avec une précision presque parfaite.


Avec torchvision.models, vous pouvez télécharger ces réseaux pré-formés et les utiliser dans vos applications. Nous allons maintenant inclure des modèles dans nos importations.
```aidl
from torchvision import datasets, transforms, models
```
La plupart des modèles pré-entraînés nécessitent que l'entrée soit des images 224x224. De plus, nous devrons faire correspondre la normalisation utilisée lorsque les modèles ont été entraînés. Chaque canal de couleur a été normalisé séparément, les moyennes sont [0,485, 0,456, 0,406] et les écarts types sont [0,229, 0,224, 0,225].


Apprentissage par transfert

La plupart du temps, vous ne voudrez pas former vous-même un réseau convolutif complet. La formation moderne des ConvNets sur d'énormes ensembles de données comme ImageNet prend des semaines sur plusieurs GPU.
 
Au lieu de cela, la plupart des gens utilisent un réseau pré-formé soit comme extracteur de fonctionnalités fixes, soit comme réseau initial à affiner.


Dans ce cahier, vous utiliserez VGGNet formé sur l'ensemble de données ImageNet en tant qu'extracteur d'entité. Vous trouverez ci-dessous un schéma de l'architecture VGGNet, avec une série de couches convolutives et une mise en commun maximale, puis trois couches entièrement connectées à la fin qui classent les 1000 classes trouvées dans la base de données ImageNet.

<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/vgg_16_architecture.png" alt="Load the Model"/>
 
 VGGNet est génial car il est simple et offre d'excellentes performances, se classant deuxième dans la compétition ImageNet. L'idée ici est de conserver toutes les couches convolutives, mais de remplacer la couche finale entièrement connectée par notre propre classificateur. De cette façon, nous pouvons utiliser VGGNet comme extracteur de fonctionnalités fixes pour nos images, puis former facilement un classificateur simple en plus de cela.
 
 - Utilisez toutes les couches, sauf les dernières, entièrement connectées comme extracteur d'entités fixes.
 - Définissez une nouvelle couche de classification finale et appliquez-la à la tâche de notre choix! Vous pouvez en savoir plus sur l'apprentissage par transfert à partir des notes de [cours CS231n Stanford](http://cs231n.github.io/transfer-learning/).


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

### Etape3 Specifies le  Loss Function et l Optimizer
Fonction d'erreur et de perte: dans la plupart des réseaux d'apprentissage, l'erreur est calculée comme la différence entre la sortie réelle et la sortie prévue.
La fonction utilisée pour calculer cette erreur est appelée fonction de perte.

#### Loss function
Les fonctions de perte sont des algorithmes mathématiques qui aident à mesurer à quel point un réseau neuronal apprend à obtenir le résultat réel. Dans l'apprentissage automatique, une fonction de perte est un algorithme mathématique qui évalue les performances d'un algorithme ML par rapport à son résultat souhaité. Il existe différentes fonctions de perte pour différents problèmes. Vous savez que le problème d'apprentissage automatique peut (en termes simples) être soit un problème de classification, soit un problème de régression. Cela implique que nous avons des fonctions de perte optimisées pour la classification et d'autres pour la régression. Pour n'en citer que quelques-uns, nous avons les fonctions de perte suivantes basées sur la classification (entropie croisée binaire, entropie croisée catégorique, similitude cosinus et autres). Nous avons également, pour n'en citer que quelques-uns, l'erreur quadratique moyenne (MSE), l'erreur en pourcentage absolue moyenne (MAPE), l'erreur absolue moyenne (MAE), pour n'en citer que quelques-uns, utilisées pour les problèmes fondés sur la régression.

#### An optimizer
Dans des phrases simples, un optimiseur peut essentiellement être appelé un algorithme qui aide un autre algorithme à atteindre ses performances maximales sans délai. En ce qui concerne l'apprentissage automatique (réseau de neurones), nous pouvons dire qu'un optimiseur est un algorithme mathématique qui aide notre fonction de perte à atteindre son point de convergence avec un délai minimum (et surtout, à réduire la possibilité d'explosion de gradient). Les exemples incluent, adam, descente de gradient stochastique (SGD), adadelta, rmsprop, adamax, adagrad, nadam etc.
Utilisez la cellule de code suivante pour spécifier une [fonctions de perte](http://pytorch.org/docs/master/nn.html#loss-functions) et un [optimiseur](http://pytorch.org/docs/master/optim.html). Enregistrez la fonction de perte choisie sous critère_transfer et l'optimiseur sous optimizer_transfer ci-dessous.

```
import torch.optim as optim
criterion_transfer = nn.CrossEntropyLoss()

optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001)
```
### Etape4 Train and Validate the Model
Entraînez et validez votre modèle dans la cellule de code ci-dessous. [Enregistrer les paramètres finaux du modèle](http://pytorch.org/docs/master/notes/serialization.html) au chemin du fichier ```'model_transfer.pt'```.
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

### Etape5 Teste le Model
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

### Etape6 Predire si la personne porte un cache nez ou non avec le  Model
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
 ### Etape7 ecrire ton Algorithm 
 
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
### Etape8 teste ta  fonction run_app 
Nous pouvons maintenant utiliser le test donne pour tester l'ensemble de données pour tester notre système.

```$xslt
## Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as you want.

for file in np.array(glob("../input/covid19-face-mask-recognition-test-data/Covid19-face-mask-recognition-test-data/*")):
    run_app(file)
```
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test result1.png" alt="result of the predition"/>
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test result2.png" alt="result of the predition"/>
<img src="https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/Resource/readme_image/test result3.png" alt="result of the predition"/>

### Etape9 integre opencv a notre project
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
 ### Etape10 Teste ton Algorithm
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

Nous avons appris à utiliser un modèle deja forme pour accélérer la formation de notre modèle et obtenir une  précision au-dessus de 80%. 

## Travail suivant 

La prochaine étape sera :
- d'intégrer le modèle avec des caméras de surveillance pour pouvoir identifier en temps réel si un homme porte son cache nez ou non.

- Intégration de systèmes d'alarme et de notification pour pouvoir avertir les agents de sécurité lorsque le système identifie un homme qui ne porte pas son cache nez. 
## Resources 
 
Le blog officiel de  [Pytorch](https://pytorch.org/blog/)
 
La documentation  officiel [Pytorch](https://pytorch.org/docs/stable/index.html)
 
Le github repository  officiel [Pytorch](https://github.com/pytorch) 
 
Le Tutoriel officiel de [Pytorch](https://pytorch.org/tutorials/)
 
le blog officiel de [pytorch](https://pytorch.org/blog/)
 
Le code peut être trouvé [ici](https://github.com/achilep/Tutoriel-Sur-Un-System-De-Reconnaissance-Du-Cache-Nez-Avec-pytorch/blob/main/systeme-de-reconnaissance-du-cache-nez.ipynb)

La version Anglaise du tutoriel peut être trouvée [ici](https://github.com/achilep/Face-Mask-Recognition-System-Tutorial-With-Pytorch) 

###### Merci pour votre attention!!!