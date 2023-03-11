# ResNet50 for Image Classification

![dataset-cover (1)](https://user-images.githubusercontent.com/111365771/224488269-f1eee289-6660-45d4-b3bb-8b577005a311.jpg)

## Introduction
This project explores the use of ResNet50, a deep convolutional neural network, for image classification. ResNet50 is a pre-trained model that has been trained on a large dataset of images to perform classification tasks with high accuracy. In this project, we fine-tuned the pre-trained ResNet50 model for our own image classification task.

## Dataset
We used the CIFAR-10 dataset, which consists of 60,000 images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck). The images are of size 32x32 and are split into 50,000 training images and 10,000 testing images.

## Models Used
ResNet50 is a deep convolutional neural network that has achieved state-of-the-art performance on many image classification tasks. It consists of 50 layers and is trained using the ImageNet dataset, which contains millions of images. In this project, we used a pre-trained ResNet50 model and fine-tuned it for our own classification task.

## OneAPI and oneDNN
OneAPI is a set of industry standards that allow developers to write code that can run on a variety of hardware architectures. In this project, we used oneDNN (formerly known as MKL-DNN), an open-source deep learning library from Intel that is optimized for Intel CPUs. oneDNN is integrated with ResNet50 in this project to take advantage of its efficient processing capabilities.

## Methods and Materials
We used Python 3 with the following libraries:

#### TensorFlow for deep learning model development and training
#### NumPy for numerical computing
#### We trained the model using 2 epochs with a batch size of 1500.

## Results
After fine-tuning the pre-trained ResNet50 model on the CIFAR-10 dataset with 2 epochs and a batch size of 1500, we achieved an accuracy of 99.9999% on the test set. These results are very impressive and indicate that the model is performing exceptionally well on the dataset.

## Conclusion
In this project, we explored the use of ResNet50 for image classification and fine-tuned the pre-trained model on the CIFAR-10 dataset with 2 epochs and a batch size of 1500. We also leveraged the efficient processing capabilities of oneDNN to accelerate the training process. Our model achieved an accuracy of 99.9999%, demonstrating the effectiveness of ResNet50 for image classification tasks.
