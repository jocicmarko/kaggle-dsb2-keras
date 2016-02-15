# Keras Deep Learning Tutorial for Kaggle 2nd Annual Data Science Bowl

This tutorial shows how to use Keras library (runs on Theano/Tensorflow backends)
to build deep neural network for
Kaggle 2nd Annual Data Science Bowl competition. This neural net achieves
~0.0359 CRPS score on the validation set.

Please note that hyper-parameters were chosen "ad-hoc", which means that
there is a lot of space for improving the score. Also, with Keras library,
it is very easy to experiment with various architectures and
hyper-parameters, so this tutorial could be a good starting point for
such experimentation.

---
## Basic info

### Data set

Raw data set for this competition provided by Kaggle is pre-processed by
```data.py``` script, where all DICOM images are resized to 64 x 64, and put together
as a time series of 30 images. Then, these input images are saved to numpy binary file
(.npy), so that they can be loaded quickly for later training.

### Model

The provided model is a convolutional neural network
(with max-pooling and dropout), and only 1 output neuron. In the input layer,
sample-wise center and std. normalization was used as a custom activation function.
Also some L2 regularization (weight decay) was used on the last fully connected layer.
The task of the net is basically a linear regression (predicting the real values
in range \[0,599\]). Linear regression was chosen since it seems like a natural
approach to this problem, instead of transforming it to a classification task.

In this example, the model objective (loss function) is root mean squared error (RMSE)
and the optimizer is Adam.


### Training

The training of the models is done in 150 iterations (epochs).
In each iteration, both systole and diastole models are trained one epoch,
and then an estimate of CRPS is calculated for both training and test data split.
Please keep in mind that this is *just an estimation* - the score won't be the same on the
validation set when submitting the result on Kaggle, but it should be somewhat similar.

Before the training, the images are denoised with TV (total variational) filter, which
smooths the image but preserves edges (this pre-processing takes 5-10 minutes).
During the training the images are augmented by random rotations in range \[-15, 15\] degrees and
by random horizontal/vertical shifts in range \[-10%, 10%\] (of image width/height).

During the training, values of loss function on test split might oscillate a bit,
but in overall they should drop. The same goes for CRPS estimate on test split.

Also, during the training, weights for last and best iteration (lowest val. loss) are saved in HDF5 files,
so that they can be loaded later for generating a submission.

On GeForce GTX 770 GPU, time needed for whole iteration
(augmenting data + training both models + evaluating CRPS) takes around 3 minutes.

### Note on CDF

In order to calculate CDF (needed for estimating CRPS and also for generating submission),
outputs from neural network are used (real values), along with parameter *sigma*.
This parameter should indicate the measure of uncertainty of the model, so that
this uncertainty is incorporated in the result. In this example, *sigma*
is chosen to be just a loss function value (RMSE). So, if the loss function is changed,
the *sigma* should be changed accordingly. Check ```utils.py``` for CDF and CRPS calculations.

---

## How to use

### Dependencies

This example depends on the following libraries:

* numpy
* scipy
* pydicom
* scikit-image
* Theano and/or Tensorflow
* Keras (0.3.1)

Also, this code should be compatible with Python versions 2.7-3.5.

### Pre-process the raw data set

In order to extract the raw data (provided by Kaggle) and save it to *.npy* files,
you should first prepare its structure. Make sure that ```data``` dir is located in the root of
this project.
Also, the tree of ```data``` dir must be like:

```
-data
 |
 ---- sample_submission_validate.csv
 |
 ---- train.csv
 |
 ---- train
 |    |
 |    ---- 0
 |    |
 |    ---- …
 |
 ---- validate
      |
      ---- 501
      |
      ---- …
```

* Now run ```python data.py```.

Running this script will create training and validation data, resize it to 64 x 64 and save to *.npy* files.

### Define the model

* Check out ```get_model()``` in ```model.py``` to modify the model, optimizer and loss function.

### Train the models

* Run ```python train.py``` to train the models.

Check out ```train()``` to modify the number of iterations (epochs), batch size, etc.

### Generate submission

* Run ```python submission.py``` to generate the submission file ```submission.csv``` for the trained model.

Check out function ```submission()``` for details. In this example, weights from best iteration
(lowest val. loss function value) are loaded and used.

---

## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
