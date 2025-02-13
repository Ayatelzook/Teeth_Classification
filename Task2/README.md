# Teeth Classification Using Transfer Learning with CNN

This project implements a Convolutional Neural Network (CNN) using transfer learning to classify images of teeth. The model is trained on a dataset of teeth images and can predict the type of teeth based on the input image. Transfer learning allows us to leverage pre-trained models, such as MobileNet and Inception, for improved accuracy and reduced training time.

## Overview
![Figure_1](<img width="1379" alt="05-transfer-learning-feature-extraction-vs-fine-tuning" src="https://github.com/user-attachments/assets/73d12914-9ec2-449e-ac57-8a5939161942" />)

The code performs the following tasks:

- **Data Preparation:** Loads and preprocesses teeth images.
- **Data Augmentation:** Applies transformations to enhance the training dataset (if applicable).
- **Model Training:** Trains a CNN on the processed images using pre-trained models.
- **Model Evaluation:** Evaluates the model on validation and test data.
- **Visualization:** Displays training results such as loss and accuracy.

## Freezing and Unfreezing Layers

In transfer learning, we often freeze the layers of the pre-trained model to prevent them from being updated during training. This allows us to retain the learned features from the initial training on a larger dataset.

### Freezing Layers

Initially, all layers of the MobileNet (or Inception) model are frozen, which means their weights will not be updated during training:

## Accuracy & loss through Training model through each epoch for MobileNet
![Figure_2]([mobile_acc](https://github.com/user-attachments/assets/098511d3-fb1d-4945-917d-6bb78b52d44d))

![Figure_3](![mobile_loss](https://github.com/user-attachments/assets/9e6d5091-6c2c-482d-8a94-3f61357c099d))


## Accuracy & loss through Training model through each epoch for Inception
![Figure_4](![inception_Acc](https://github.com/user-attachments/assets/5f9ef203-e239-41a7-b228-967f62a4dd7d)
)

![Figure_5](![inception_loss](https://github.com/user-attachments/assets/866eb2c3-9728-411c-a7fe-79a7338b63cb)
)

- ## Results of prediction from streamlit
 ![Figure_6](![gum](https://github.com/user-attachments/assets/a5d9af08-148f-48d2-9df5-f62a33bbe1d6)
)

![Figure_7](![inception](https://github.com/user-attachments/assets/8fac386b-77dd-4ff7-9283-df87b6a98742)
) 
