In this project we'll attempt to train a neural network to land a rocket [à la SpaceX](https://www.youtube.com/watch?v=u0-pfzKbh2k) in a simple physics simulation. We'll add complexity to the problem as we progress. Please read through the below for an overview of the ideas. 

#### Download the materials with [this link](https://github.com/conor-or/ai-snake/archive/master.zip)

#### You will need to install [TensorFlow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/#installation) (in that order) before we start.

# 1. Some Rocket Science

A rocket moves around simply 

# 2. Neural Networks

A neural network (NN) is essentially just a very complicated function which takes many inputs and produces (usually) one or a small number of outputs. It's used to make predictions based on lots of input data where the functional form of the predictor itself is not clear. For example, classifying the objects in an image (input: 512x512 grid of pixel values, output: text label describing the object).

![NN](https://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg)

The NN is made up of nodes or _neurons_ organised into _layers_. Each neuron is really a very simple function (for example: _y_ = _w_ * _X_). It receives some _X_ from the input data and computes _y_ given some constant(s) _w_ called the _weights_. If our input has 4 numbers then all the nodes in the first _layer_ take a vector _X_ of 4 numbers. These nodes might connect to one single node providing an output _y_ or they might again be the input for a second layer. A NN with multiple layers between input and output is called a _deep neural network_ and the layers between input and output are called _hidden layers_.

The __weights__, _w_ tell the network exactly what output to give based on the input and the weights encode all of the information about the data we have. We learn the weights by _training_ the network. At first the weights are random. We then use the network to predict a _y_ value based on some data where we _already know the true outcome_. The network then adjusts its own weights depending on how close it was to the true value. This data, where we already know the true outcome to compare to the prediction, is called the _training data_ or _training set_.

# 3. Q Learning

The training process becomes more difficult when the outputs of the network do not produce an immediate feedback that it can train on.
