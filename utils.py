import os
import sys
import heapq
import operator
import pickle
from collections import Counter, OrderedDict, defaultdict
import traceback
import os
import h5py
import sys
import datetime
import time
import random
import numpy as np
# import matplotlib.pyplot as plt
# import cleverhans
# from cleverhans.attacks import SaliencyMapMethod, FastGradientMethod, CarliniWagnerL2, BasicIterativeMethod
# from cleverhans.utils_keras import KerasModelWrapper
from tensorflow.keras import backend as K
# from keras.applications.imagenet_utils import preprocess_input
# from keras.datasets import mnist, cifar10
# from keras.preprocessing import image
# from keras.models import model_from_json
# from keras.layers import Input
# from keras.utils import np_utils
from tensorflow.keras import models
# from lrp_toolbox.model_io import read
random.seed(123)
np.random.seed(123)

import numpy as np
import pandas as pd

# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
def calc_major_func_regions(model, train_inputs, skip=None):
    if skip is None:
        skip = []

    outs = get_layer_outs_new(model, train_inputs, skip=skip)

    major_regions = []

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        layer_out = layer_out.mean(axis=tuple(i for i in range(1, layer_out.ndim - 1)))

        major_regions.append((layer_out.min(axis=0), layer_out.max(axis=0)))

    return major_regions
def find_shared_neurons(listA, listB):
    """
    :param listA: list of unique neurons in A(dtype:list of int)
    :param listA: list of unique neurons in B(dtype:list of int)
    :return: shared neurons between list A and B(dtype:list of int)
    """
    shared_neurons = set.intersection(set(listA), set(listB))
    return list(shared_neurons)

def get_layer_outs_new(model, inputs, skip=[]):

    # skip.append(0)
    evaluater = models.Model(inputs=model.input,
                             outputs=[layer.output for index, layer in enumerate(model.layers) \
                                      if index not in skip])

    # Insert some dummy value in the beginning to avoid messing with layer index
    # arrangements in the main flow
    # outs = evaluater.predict(inputs)
    # outs.insert(0, inputs)

    # return outs

    return evaluater.predict(inputs)
def percent_str(part, whole):
    return "{0}%".format(float(part) / whole * 100)
def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin

    return X_scaled
def get_trainable_layers(model):
    trainable_layers = []
    for idx, layer in enumerate(model.layers):
        try:
            if 'input' not in layer.name and 'softmax' not in layer.name and \
                    'pred' not in layer.name and 'drop' not in layer.name:
                weights = layer.get_weights()[0]
                trainable_layers.append(model.layers.index(layer))
        except:
            pass

    trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return trainable_layers
def get_layer_inputs(model, test_input, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_input)

    inputs = []

    for i in range(len(outs)):
        weights, biases = model.layers[i].get_weights()

        inputs_for_layer = []

        for input_index in range(len(test_input)):
            inputs_for_layer.append(
                np.add(np.dot(outs[i - 1][0][input_index] if i > 0 else test_input[input_index], weights), biases))

        inputs.append(inputs_for_layer)

    return [inputs[i] for i in range(len(inputs)) if i not in skip]
def get_layer_outs(model, test_input, skip=[]):
    inp = model.input  # input placeholder
    outputs = [layer.output for index, layer in enumerate(model.layers) \
               if index not in skip]

    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layer_outs = [func([test_input]) for func in functors]

    return layer_outs
def get_dense_layers(model):
    dense_layers = []
    for idx, layer in enumerate(model.layers):
        try:
            if 'dense'  in layer.name :
                weights = layer.get_weights()[0]
                dense_layers.append(model.layers.index(layer))
        except:
            pass

    # trainable_layers = trainable_layers[:-1]  # ignore the output layer

    return dense_layers
def generate_adversarial(original_input, method, model,
                         target=None, target_class=None, sess=None, **kwargs):
    if not hasattr(generate_adversarial, "attack_types"):
        generate_adversarial.attack_types = {
            'fgsm': FastGradientMethod,
            'jsma': SaliencyMapMethod,
            'cw': CarliniWagnerL2,
            'bim': BasicIterativeMethod
        }

    if sess is None:
        sess = K.get_session()

    if method in generate_adversarial.attack_types:
        attacker = generate_adversarial.attack_types[method](KerasModelWrapper(model), sess)
    else:
        raise Exception("Method not supported")

    if type(original_input) is list:
        original_input = np.asarray(original_input)
    else:
        original_input = np.asarray([original_input])

        if target_class is not None:
            target_class = [target_class]

    if target is None and target_class is not None:
        target = np.zeros((len(target_class), model.output_shape[1]))
        target[np.arange(len(target_class)), target_class] = 1

    if target is not None:
        kwargs['y_target'] = target

    return attacker.generate_np(original_input, **kwargs)
def filter_correct_classifications(model, X, Y):
    X_corr = []
    Y_corr = []
    X_misc = []
    Y_misc = []
    preds = model.predict(X)  # np.expand_dims(x,axis=0))

    for idx, pred in enumerate(preds):
        if np.argmax(pred) == np.argmax(Y[idx]):
            X_corr.append(X[idx])
            Y_corr.append(Y[idx])
        else:
            X_misc.append(X[idx])
            Y_misc.append(Y[idx])

    '''
    for x, y in zip(X, Y):
        if np.argmax(p) == np.argmax(y):
            X_corr.append(x)
            Y_corr.append(y)
        else:
            X_misc.append(x)
            Y_misc.append(y)
    '''

    return np.array(X_corr), np.array(Y_corr), np.array(X_misc), np.array(Y_misc)
def filter_val_set(desired_class, X, Y):
    """
    Filter the given sets and return only those that match the desired_class value
    :param desired_class:
    :param X:
    :param Y:
    :return:
    """
    X_class = []
    Y_class = []
    for x, y in zip(X, Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)
    print("Validation set filtered for desired class: " + str(desired_class))
    return np.array(X_class), np.array(Y_class)