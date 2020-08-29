# Dog breed classifier
###### Udacity project: Dog breed classifier
###### Author: Stefano Medagli
###### date: 30.07.2020
###### ver: 0.1

## Scope
The models accepts an image as input.
<br>
If a dog is detected in the image, it will provide the estimated dog's breed.
<br>
If a human is detected, it will provide an estimate of the dog breed that is most resembling.
<br>
If neither a human nor a dog is detected in the image, will return an error message.
<br>
The algorithm(s) can be used as part of a mobile or web app.
<br>
Models are in general of 2 kinds:
* CNN from the sketch
* Transfer learning model

### CNN from the sketch
The default CNN network is a sequential network with the following layers:

    1) Conv2D
    2) MaxPool2D
    3) Conv2D
    4) MaxPool2D
    5) GlobalAveragePooling2D
    6) Dropout
    7) Dense with 133 nodes (since there are 133 possible dog breeds) - Output

To edit this architecture see *dog_breed.models.build_network.py*.
<br>

Because of the architecture and the small training sample,
performance of this kind of CNN will be pretty poor.
 
### Transfer learning
Transfer learning considerably improves performance.
<br>
When using transfer learning,
the network will "exploit" _bottleneck features_ from pre-trained network.
<br>
The possible pre-trained network are: 
* VGG16
* VGG19
* Resnet50
* InceptionV3
* Xception

Once computed the bottleneck features,
they will feed a network constituted by 2 layers:

    1) GlobalAveragePooling2D
    2) Dense with 133 nodes (since there are 133 possible dog breeds) - Output

To edit this architecture see *dog_breed.models.build_network.py*.

## prerequisites
create a conda environment using the file in `environment/pkg.txt`

```bash
conda create --name breed --file dog_breed/environment/pkg.txt
```
To run correcly the dog classifier it is necessary to add the module folder to the environmental paths.
. 
#### @TODO:
    - data augmentation for transfer learning?
    - write dog_breed.models.train_and_predict headline and explanation in the Readme
    - refactor code
    - refactor functions that have too many arguments)
    - resolve internal "import" instructions in bottleneck_features.py
    - mkdir when saving files in non existing folders
    - version that does not save in memory temporary files
## folder structure
the code was written considering the following folder structure.
<br>
Different architecture may produce error and need to adapt the paths

```bash
|   .gitignore
|   __init__.py
|   README.md
|
+---common
|   |   __init__.py
|   |   graph.py
|   |   metrics.py
|   |   paths.py
|   |   tools.py
|       
+---data
|   |   __init__.py
|   |   analysis.py
|   |   datasets.py
|   |
|   +---bottleneck_features
|
|   +---dogImages
|
|   +---saved_models
|
+---detectors
|   |   __init__.py
|   |  detectors.py
|   |
|   +---saved_detectors
|   |    |    haarcascade_frontalface_alt.xml
|
+---environment
|   |   pkg.txt
|   |   tree.txt
|       
+---images
|
+---models
|   |   __init__.py
|   |   bottleneck_features.py
|   |   build_network.py
|   |
|   +---cnn
|   |    |   __init__.py
|   |    |   train_and_predict_cnn.py
|   |    |   train_multiple_cnn.py
|   |
|   +---tl
|   |    |   __init__.py
|   |    |   train_and_predict_tl.py
|   |    |   train_multiple_tl.py
|   |    |   transfer_learning.py   
|   |     
|       
+---notebooks
|   |   data_analysis.ipynb
|
+---preprocessing
|   |   __init__.py
|   |   preprocess.py
|
+---samples
|   |   sample_dog.jpg
|   |   sample_human_2.png
|
+---scripts
|   |   dog_detector.py
|   |   evaluate.py
|   |   human_detector.py
|   |   transfer_learning_evaluate.py
|   |   transfer_learning_train.py
|
```

## components
#### common
* *graph.py*:
contains the common arguments for graphical objects
* *metrics.py*:
contains the different metrics to evaluate the quality of predictions
* *models_param.py*:
contains common functions and parameters for model training
* *paths.py*:
the module defines the default paths and functions to navigate the project
* *tools.py*:
This module contains common tools

#### data
* *analysis.py*:
contains functions and methods to an initial data analysis
* *datasets.py*:
contains the functions to handle training, validation and test datasets

#### detectors
* *detectors.py*:
implements the human and dog detector for images

#### models
* *bottleneck_features.py*:
contains functions to compute the bottleneck features for pre-trained networks
* *build_network.py*:
contains the function(s) to create the networks to implement transfer learning and a CNN from the sketch.
* *dog_classifier.py*:
contains functions for final dog classification tasks
#### cnn
contains tools and functions to handle the case of CNN from the sketch
* *train_and_predict_cnn.py*:

* *train_multiple_cnn.py*:
contains methods to train multiple CNNs from sketch
#### tl
contains tools and functions to handle the case of Transfer learning (need refactoring of the code)
* *train_and_predict_tl.py*:

* *train_multiple_tl.py*:
contains methods to use transfer learning to train multiple models
* *transfer_learning.py*:
contains the tool to generate and evaluate models using transfer learning
#### preprocessing
* *preprocess.py*:
contains tools for preprocessing of the images

## Scripts
#### dog_classifier.py
Will print the dog breed
(or the breed to which the human in the picture resembles the most.)
```bash
python scripts/dog_detector.py -f <path_to_image>
```
#### dog_detector.py
Will print whether or not there is a dog in the input image file.
###### example
```bash
python scripts/dog_detector.py -f <path>
```
#### evaluate.py
Evaluates the performance (in terms of test accuracy) of the saved models.
<br>
Iterates through the .hdf5 files of a specified folder and assigns the weights to the basic transfer learning network.
<br>
Then computes the test accuracy for each of them and reports it into _report.csv_ file.
###### example
```bash
python scripts/evaluate.py -m <models_folder> -o <output_folder>
```
#### human_detector.py
Will print whether or not there is a human face in the input image file.
###### example
```bash
python scripts/human_detector.py -f <path>
```
#### transfer_learning_evaluate.py
Uses transfer learning to evaluate performance of a network.
###### example
```bash
python scripts/transfer_learning_evaluate.py -n <pre-trained_network>
```
other options are:
* -e: set number of epochs
* -a: use data augmentation
* -p: change prefix of the weights' file
* -o: overwrite existing weights' file
* -tr: evaluate performance on training dataset
* -vl: evaluate performance on validation dataset
#### transfer_learning_train.py
Uses transfer learning to train a network.
###### example
```bash
python scripts/transfer_learning_train.py -n <pre-trained_network>
```
other options are:
* -e: set number of epochs
* -a: use data augmentation
* -p: change prefix of the weights' file
* -o: overwrite existing weights' file
