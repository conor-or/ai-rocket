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

# 1. Q-Learning

Any machine learning method can be reduced to an answer to the following question: 'given some input __X__, how do we maximize some other function _f(___X___)_?
