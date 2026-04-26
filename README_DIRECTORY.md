# README Overview

This repository contains several README files, each covering a different aspect of the project.

## READMEs from Sam3 Installation

- Sam3_README.md: Main documentation for the SAM3 model, including its capabilities, architecture, installation, usage, and related resources.

- README_TRAIN.md: Instructions for training and fine-tuning SAM3 models on custom datasets, including installation, usage examples, and configuration. This is not necessary to run our project code, but it was included with the Sam3 installation.

- CODE_OF_CONDUCT.md: Guidelines for community behavior and participation standards. Not necessary for our code.

- CONTRIBUTING.md: Information on contributing to the project, including pull request guidelines and contributor license agreements. Not necessary for our code.

## READMEs for Our Code

All attack- and defense-related code is located in Adversarial_attack_code.

- Adversarial_attack_code/HOW_TO_RUN.md: Instructions for running the adversarial attack scripts in the subfolder.

- Adversarial_attack_code/ADVERSARIAL_ATTACKS_README.md: Documentation for the adversarial attack framework on SAM3, including supported attack methods and usage examples.

- Adversarial_attack_code/DEFENSE_README.md: Guide for running preprocessing-based defenses against adversarial attacks.

- Adversarial_attack_code/RESNET_ADVERSARIAL_README.md: Documentation for the ResNet-18 adversarial attack framework, including requirements and usage.

## Running Animal Classification and Segmentation Scripts

To run the animal classification script using ResNet:
```
python run_animal_classification_resnet.py
```
Results are saved in the output directory: willtommie-sam3/animal_results_resnet

To run the animal segmentation script:
```
python run_animal_segmentation.py
```
Results are saved in the output directory: willtommie-sam3/animal_results