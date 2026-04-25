# README Overview

This repository contains several README files, each providing specific information about different aspects of the project. Below is a brief introduction to each:

*READMEs from Sam3 installation*

- **Sam3_README.md**: The main documentation for the SAM 3 model, including its capabilities, architecture, installation, usage, and links to related resources.

- **README_TRAIN.md**: Instructions for training and fine-tuning SAM3 models on custom datasets, including installation, usage examples, and configuration. This is not neccesary to run our project code, but it did come with the Sam3 instalation.

- **CODE_OF_CONDUCT.md**: Guidelines for community behavior and standards for participation in the project. Again, not neccesary for our code.

- **CONTRIBUTING.md**: Information on how to contribute to the project, including pull request guidelines and contributor license agreement. Not neccesary for our code

--------------------------------------------------------

*READMEs for our code*

All the attack and defense related code is in Adversarial_attack_code

- **Adversarial_attack_code/HOW_TO_RUN.md**: Instructions on how to run the adversarial attack scripts from the subfolder.

- **Adversarial_attack_code/ADVERSARIAL_ATTACKS_README.md**: Documentation for the adversarial attack framework on SAM3, including supported attack methods and usage examples.

- **Adversarial_attack_code/DEFENSE_README.md**: Guide for running preprocessing-based defenses against adversarial attacks on models.


- **Adversarial_attack_code/RESNET_ADVERSARIAL_README.md**: Documentation for the ResNet-18 adversarial attack framework, including requirements and usage.


## Running Animal Classification and Segmentation Scripts

To run the animal classification script using ResNet:
```
python run_animal_classification_resnet.py
```

To run the animal segmentation script:
```
python run_animal_segmentation.py
```