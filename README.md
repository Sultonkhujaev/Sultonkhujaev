- 👋 Hi, I’m Izzatillakhon 
- 👀 I’m interested in Data science
- 🌱 I’m currently learning Computer Science
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me telegram: izzatillakhon1@gmail.com

<!---
Sultonkhujaev/Sultonkhujaev is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
Hi there 👋
This is my current project
Face mask detection with Pytorch

My current project

🔭 I’m currently working on Face mask detection APP
🌱 I’m currently learning Docker, Kubernetes
👯 I’m looking to collaborate on MedTech software
🤔 I’m looking for help with Data annotation
💬 Ask me about Everything
📫 How to reach me: shushukurov@gmail.com
😄 Pronouns: Shakhzod 3 times :)
⚡ Fun fact: I am a powerlifting champion 🏋🏻🏆 -->
Table of contest
1. ML portfolio
List of My projects Inspired by Andrej Karpathy, Justin Johnson, Cs231n, Elon Musk's tweet about PyTorch :) Most projects are implented on low level using Pytorch tensors only as a gpu accelerating data type, only few of them (most complicated ones) utilize some pytorch high level API functions

1.1 KNN for Image Classification
KNN classifier for CIFAR-10 from scratch with PyTorch. KNN is data driven, image classification algorithm that was popular before Deep Learning came out. So I structured my PyTorch porfolio projects according to timeline of algorithms were developed (Popular) So I have started with KNN

1.2 Linear Classifiers for image classification
Overview. I have developed a more powerful approach to image classification than KNN that will eventually naturally extend to entire Neural Networks and Convolutional Neural Networks. The approach has two major components: a score function that maps the raw data to class scores, and a loss function that quantifies the agreement between the predicted scores and the ground truth labels. It then casts this as an optimization problem in which minimizes the loss function with respect to the parameters of the score function. (Pretty much same to any deep learning algorithms). Here i also introducet the idea of regularization (L1, L2, Elastic net (L1+L2)), In next projects I added (Dropout and Batch Norm)

Support Vector Machine
The SVM loss is set up so that the SVM “wants” the correct class for each image to a have a score higher than the incorrect classes by some fixed margin Δ. (Uses hinge loss)

Softmax
the Softmax classifier which has a different loss function than SVM. If you’ve heard of the binary Logistic Regression classifier before, the Softmax classifier is its generalization to multiple classes. In the Softmax classifier, the function mapping f(xi;W)=Wxi stays unchanged, but now it is interpreted these scores as the unnormalized log probabilities for each class and replace the hinge loss with a cross-entropy loss.

1.3 Deep Learning Modules + Functions implementation
This Project consists of Modular implementation of Fully-Connected Neural Networks, Dropout and different optimizers (SGD, Momentum, Rmsprop, Adam)

Modular implementation of Fully-Connected Neural Networks
Andrej Karphathy once wrote that ML engineers should have deep understanding of backpropagation. Therefore I implemented all necessary modules for Neural Networks from scratch using PyTorch GPU acceleration in order Improve knowledge of NN and Backprop.

Dropout
Dropout is a technique for regularizing neural networks by randomly setting some output activations to zero during the forward pass.

Optimizers (SGD, Momentum, Rmsprop, Adam)
So far I have used vanilla stochastic gradient descent (SGD) as my update rule. More sophisticated update rules can make it easier to train deep networks. Therefore I have implement a few of the most commonly used update rules (SGD, Momentum, RMsprop, Adam) and compare them to vanilla SGD.

1.4 Convolutional Neural Network, Batch Normalization and Kaiming initialization
In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery

Batch Norm
Batch normalization (also known as batch norm) is a method used to make artificial neural networks faster and more stable through normalization of the input layer by re-centering and re-scaling. It was proposed by Sergey Ioffe and Christian Szegedy in 2015.

Kaiming Initialization
Kaiming Initialization, or He Initialization, is an initialization method for neural networks that takes into account the non-linearity of activation functions, such as ReLU activations.

1.5 Image Captioning (RNN, LSTM, Attention)
Generally, a captioning model is a combination of two separate architecture that is CNN (Convolutional Neural Networks)& RNN (Recurrent Neural Networks) and in this case LSTM (Long Short Term Memory), which is a special kind of RNN that includes a memory cell, in order to maintain the information for a longer period.

Recurrent Neural Network
Recurrent Neural Network (RNN) language models for image captioning.

Long-Short-Term-Memory
Many people use a variant on the vanilla RNN called Long-Short Term Memory (LSTM) RNNs because Vanilla RNNs can be tough to train on long sequences due to vanishing and exploding gradients caused by repeated matrix multiplication. LSTMs solve this problem by replacing the simple update rule of the vanilla RNN with a gating mechanism as follows.

Attention
Attention is the idea of freeing the encoder-decoder architecture from the fixed-length internal representation. This is achieved by keeping the intermediate outputs from the encoder LSTM from each step of the input sequence and training the model to learn to pay selective attention to these inputs and relate them to items in the output sequence.

1.6 Neural Networks Visualization
Saliency Maps
A saliency map tells the degree to which each pixel in the image affects the classification score for that image.

Adversarial Attacks
Use of image gradients to generate "adversarial attacks". Given an image and a target class, It is possible to perform gradient ascent over the image to maximize the target class, stopping when the network classifies the image as the target class.

Class visualization
By starting with a random noise image and performing gradient ascent on a target class, It is possible to generate an image that the network will recognize as the target class.

1.7 Style Transfer (Soon)
Neural Style Transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation.

1.8 Single-Stage Object Detector (YOLO 2)
In project I implemented a single-stage object detector, based on YOLO (v1 and v2) and used it to train a model that can detect objects on novel images. I also evaluated the detection accuracy using the classic metric mean Average Precision (mAP). In Next project (That extends this project), I will implement a two-stage object detector, based on Faster R-CNN. The main difference between the two is that single-stage detectors perform region proposal and classification simultaneously while two-stage detectors have them decoupled.

1.9 Two-Stage Object Detector (Faster R-CNN)
In this project I implemented a two-stage object detector, based on Faster R-CNN, which consists of two modules, Region Proposal Networks (RPN) and Fast R-CNN and extends previous project YOLO detector. I use it to train a model that can detect objects on novel images and evaluate the detection accuracy using the classic metric mean Average Precision (mAP)

1.10 Resnet (Pre-Resnet + BottleNeck block)
This project is motivated by my desire to deeply understand the state of art architecutre 'Residual network' AKA 'Resnet'. When I was studying CNNs for visual recognition I always tried to understand papers first then re-implement by myself using favourite tools e.g Pytorch or Numpy. However, I could not fint any tutorial to re-implement Resnet based architures from scratch using pytorch (All I could find using PyTorch Implemented models from Model Zoo) So I decided to try myself mainly by looking on CS231n lecture notes (Mainly raw NumPy based code) and re-implent it using only PyTorch

1.11 Generative Adversarial Network
(Soon)
