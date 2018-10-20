# Neural-Networks-Machine-Learning-Examples

## Introduction

These are a set of codes examples from my course Neuro Control Motor in 2017. The idea is that people who are staring studying artificial Neural Networks (ANN) or Machine Learning Algorithms could have a view of what you can achieve with this algorithms.

The codes were written in Matlab R2015a. I will try to create the Python version later. Some of the variables and comments could be also in Spanish. However, it's nothing that Google Translator cannot solve XD.

## Description of files:

### SOM_v1.m - Self Organized Map

This example shows the Self Organized Map or SOM neural network in action. In summary, this type of ANN is trained using Unsupervise Learning to generate a representation of an input space, in this case a two dimensional equilateral tiangle (Fig. 1.).

![image](https://user-images.githubusercontent.com/15948497/47254783-381eeb00-d42c-11e8-8109-7d9229022640.png)
Figure 1.

Before starting the training process of this SOM, their neurons (Fig. 2, blue points) occupy a random position compared with the input space (Fig. 2, red points)

![image](https://user-images.githubusercontent.com/15948497/47254471-bb3e4200-d428-11e8-85c3-b1b68948439a.png)
Figure 2.

After giving the input space to the SOM 500 times, the neurons have learn to represent the topology of the input space as shown in (Fig. 3, blue points)

![image](https://user-images.githubusercontent.com/15948497/47254677-29840400-d42b-11e8-971c-67adeb52351d.png)
Figure 3.

### RBF_v1.m - Radial Basis Function Network

The RBF or Radial Basis Function Neural Network is a type of NN which structure is composed by one hidden layer of neurons with radial (gaussean) activation function, and an output layer  with neurons that performs lineal activation functions. 

This type of neural network is tipically used for an interpolation approach. In this example we have an input space with four stablish categories (Fig. 1.)

![image](https://user-images.githubusercontent.com/15948497/47255801-d23a5f80-d43b-11e8-9dde-ea99b7201451.png)

Figure 1.

The outcome of this network after training is to approximate its hidden neurons (also called centroids) to the input space as shown in (Fig. 2)

![image](https://user-images.githubusercontent.com/15948497/47255855-568ce280-d43c-11e8-9204-f231b283e6ab.png)
Figure 2.
