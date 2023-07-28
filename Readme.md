
# DeepKnowledge: Knowledge-Driven automated testing of Deep-Neural Networks
###### This is the SESAME Assurance tool-supported technique of Data-Driven and Learning Components of EDDIs


This code is an implementation of our tool-supported technique DEEPKNOWLEDGE.
It is in beta version and provided as a reference to the approach presented in deliverable 6.1. 
If you encounter any problems, do not hesitate to reach out to the York team (sondess.missaoui@york.ac.uk).

Scripts are tested with the open-source machine learning framework Keras (v2.2.2) with Tensorflow (v2.6) backend.

## Abstract
The increasing use of Deep Neural Networks (DNNs) in safety- and security-critical applications created an urgent need for effective 
DNN testing techniques capable of establishing the dependability levels of DNNs.
Despite noteworthy advances, recent research proposes testing techniques and coverage criteria that focus primarily on the analysis of the DNN 
behaviour using the conventional training-validation-testing DNN practice, often neglecting important generalisation capabilities exhibited by DNNs. 
We address this gap by introducing DEEPKNOWLEDGE, a systematic testing methodology for DNNs founded on the principle of knowledge transfer. 
Our empirical evaluation of several DNNs, across multiple datasets and state-of-the-art adversarial generation techniques demonstrates 
the usefulness and effectiveness of DEEPKNOWLEDGE and its ability to support the engineering of more dependable DNNs.
DEEPKNOWLEDGE provides to our partners an easy and effective assurance technique for their Machine Learning componenets.
our tool in a simple multi-paradigm technology, i.e., using a set of Python scripts. 
This enables a quick adaptation and code maintenance by our partners. 
The developed prototype tool is also compatible with most systems and architectures used by our industrial partners in SESAME.

## Install Required Packages
We recommend to start by creating a virtual environement then install the required packages.

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
    
```
pip install tensorflow==2.8.0
pip install numpy==1.21.5

pip install keras==2.8.0 
pip install protobuf==3.20.1

pip install numpy==1.21.5
pip install pandas==1.3.5
pip install sklearn

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
