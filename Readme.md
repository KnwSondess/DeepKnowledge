
# DeepKnowledge: 
Testing Deep Neural Networks Using Generalisation-based
Coverage Criterion

This code is an implementation of our tool-supported technique DEEPKNOWLEDGE.


Scripts are tested with the open-source machine learning framework Keras (v2.2.2) with Tensorflow (v2.6) backend.

## Abstract
Despite their unprecedented success, DNNs are notoriously fragile to small shifts in data distribution, 
demanding effective testing techniques capable of providing evidence for DNN dependable operation.
Despite recent advances in devising techniques for the systematic testing of DNN-based systems, there is a lack of holistic testing and quality assurance methodologies that assess DNN's capability to generalize and operate comparably beyond data included in their training distribution. 
We address this gap by introducing \approach, a systematic testing methodology for DNN-based systems founded on the theory of knowledge generalisation, which aims to enhance DNN robustness and reduce the %unreasonable 
residual risk of `black box' models. 
Conforming to this theory, \approach\ posits that core computational DNN units have the capacity to generalise under domain shift. \approach\ provides an objective confidence measurement on testing activities of DNN in any data distribution. Our empirical evaluation of several DNNs, across multiple datasets and state-of-the-art adversarial generation techniques demonstrates the usefulness and effectiveness of \approach\ and its ability to support the engineering of more dependable DNNs.
We report improvements of up to 10 percentage points over state-of-the-art coverage criteria for detecting adversarial attacks on several benchmarks, including MNIST, SVHN, and CIFAR.

## Install Required Packages
We recommend starting by creating a virtual environment and then installing the required packages.

#### Vitual Environement

```
python3 -m pip install --user virtualenv

python3 -m venv path/to/the/virtual/environment
```
###### Activate virtual environment

```
source path/to/the/virtual/environment/bin/activate
```



#### Linux
To install the required packages
```
pip install -r DeepKnw_requirements.txt

```
## Linux
To download SVHN dataset
```
cd dataset/SVHN
sudo bash svhn.sh

```
To download EMNIST dataset

```
cd dataset/emnist
sudo bash download.sh

```

## Runing DEEPKNOWLEDGE
use shell command

```
$ cd path/to/the/project/folder

$ python Coverage_Estimation.py –model [path_to_keras_model_file] –dataset svhn –approach knw –
threshold 0.5 –logfile [path_to_log_file]
```
## Parameters for configuring 
```
- model => The name of the Keras model file. Note the architecture file (i.e., JSON) and the weights
file should be saved separately as an .h5 file. If the model is trained and saved into a file, it needs to be
in the (.hdf5) format. You need also to save all the model under the same folder ‘Networks’.
- dataset = Name of the dataset to be used. Current implementation supports ‘MNIST’, ‘Cifar10’ and
‘Cifar100’, ’fashion MNIST’ and ‘SVHN’. The code is configurable and adding another is possible
through a loading function in the script ‘Dataprocessing.py’.
- approach = The approach for coverage estimation. This includes DEEPKNOWLEDGE noted as ‘knw’,
and other baselines implemented for our empirical study. Current implementation supports DeepIm-
portance ‘idc’,‘nc’,‘kmnc’,‘nbc’,‘snac’,‘tknc’,‘ssc’, ‘lsa’, and ‘dsa’.
- neurons = The number of neurons used in ‘knw’
- threshold = a threshold value used to consider if a neuron is a transfer knowledge neuron or not for
‘knw’.
- advtype = The name of the adversarial attack to be applied. This implementation supports ‘mim’,
‘bim’, ‘cw’, and ‘fgsm’ techniques.
- class = selected classes. Note this argument is used for approaches like ‘kmnc’,‘nbc’,‘snac’ and has
no use for ‘knw’ context. This argument takes a number between 0 and 9 for mist or cifar10 and svhn.
- layer = The subject layer’s index for approaches including ‘idc’,‘tknc’, ‘lsa’. Note that only trainable
layers can be selected.
- logfile = The name of the file that the results to be saved.
- repeat = Obtained results are added to repeat the experiments to reduce the randomness effect.
```
