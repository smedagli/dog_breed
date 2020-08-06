# Dog breed classifier
###### Udacity project: Dog breed classifier
###### Author: Stefano Medagli
###### date: 30.07.2020
###### ver: 0.1

## Scope
The model accepts an image as input.
<br>
If a dog is detected in the image, it will provide the estimated dog's breed.
<br>
If a human is detected, it will provide an estimate of the dog breed that is most resembling.
<br>
If neither a human nor a dog is detected in the image, will return an error message.

The algorithm(s) can be used as part of a mobile or web app.


## prerequisites
create a conda environment using the file in `environment/pkg.txt`

```bash
conda create --name breed --file dog_breed/environment/pkg.txt
```
#### @TODO:
    * update README
    * create `tree.txt` file
    * create `pkg.txt` file
    * implement human detector
	* implement dog detector
	* Create CNN(s) to classify dog's breed (using transfer learning)
	* Create script to train network
	* Create a script for prediction with arguments
		* -n trained network
		* -i input image
	* Create preprocessing module (resize of input image)
	

## folder structure
```bash
|   .gitignore
|   paths.py
|   README.md
|   __init__.py
|
+---common
|   |   paths.py
|   |   __init__.py
|       
+---data
|   |   __init__.py
|           
+---environment
|       pkg.txt
|       tree.txt
|       
+---saved_models
|
+---models
|       (model.pkl)
|       metrics.py
|       train_classifier.py
|       train_classifier_script.py
|       __init__.py
|       
```

## Steps

## pre-processing data
To preprocessing of the data.
Run the script as
```bash
python ...
```

## build/train model
To train the model, run
```bash
python ...
```

### components
#### common
* *paths.py*:
the module defines the default paths of the project