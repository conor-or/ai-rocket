In this project we'll attempt to train a neural network to land a rocket in a simple physics simulation. We'll add complexity to the problem as we progress. You'll develop lots of new Python skills as we go on, especially object-oriented programming and using some new libraries. Please read through the below for an overview of the ideas. 

# 0. Preliminaries

First you need to install two things: Keras and TensorFlow.

__TensorFlow__ is the industry standard neural network package, developed and maintained by Google. Installation instructions are [here](https://www.tensorflow.org/install/pip) but it should amount to running the command
```
$ pip install tensorflow
```
in a terminal. TF requires a lot of effort to set up for a basic neural network so we're going to use a library called __Keras__ to actually build the network. Keras is an interface for TF. It's very easy to set up simple networks in Keras but they are effectively running in TF in the background.

Once you've installed TF, install Keras as well. Instructions are [here](https://keras.io/#installation) but again it should be as simple as
```
$ pip install keras
```
Once you've installed both open Python and try to run the following code
```
>>> from keras.model import Sequential
>>> model = Sequential()
```
If those lines run without any issues the installation was successful. 

_There might be a warning about np.float types being deprecated, ignore this_.

# 1. Machine Learning in General

Any machine learning method can be reduced to an answer to the following question: 'given some input __X__, how do we maximize some other variable that is a function of __X__, i.e. _y=f(___X___)_ ? In many cases in physics we can find _f_ from some physical intuition and differentiate to find maxima. In the case of ML problems, _f_ is some incredibly complex function over hundreds or thousands of inputs that is impossible to find analytically. Rather, we take lots of __X__ for which we have the _y_ already (the training data), and use a method to approximate _f_.

Classifying images is a good example. Say we have lots of images which can each be expressed as matrices __X__ of pixel values. For each image say we also have a classification (from a human) for what the image contains. This classification could be represented as a vector __y__ with a 1 in the line corresponding to the classification. All we need then is a function __y__= _f_(__X__) that gives us the __y__ for each __X__. In this case _f_ would be a __neural network__...

# 2. Neural Networks

NNs are essentially just functions that take some input and give you an output. What goes on inside the network to get to that output is in general incredibly complex.
