# Neural Networks for Atmospheric Sciences
## Viewing forced climate patterns through an AI Lens

## This Repository consists of 

Neural Network Code for the paper 'Viewing Forced Climate Patterns through an AI Lens'
by **Elizabeth A. Barnes**, **James W. Hurrell**, **Imme Ebert-Uphoff**, **Chuck Anderson**, and **David Anderson**


### Prerequisites

Python 3.7 default libraries
copy
system
time
math
matplotlib.pyplot

Additional libraries:
NumPy
Pytorch

```
conda install -c anaconda numpy 
pip install numpy

conda install pytorch -c pytorch
pip install torch
```

### Importing and usage

Import the neuralnetworks.py


```
import neuralnetworks as nn
```

Define and create a neural net object

```
nn.NeuralNetwork?
NeuralNetwork(n_inputs, n_hiddens_list, n_outputs)

ni = 1 #Network input shape
nhs = [5, 5] #List of values for units per hidden layer
no = 1 #Number of outputs

nnet = nn.NeuralNetwork(ni, nhs, no)
```

## Training and Running a Model

After the network object has been created, you can train your network model using the .train() function. Inputs (X_array) and targets (T_array) should be numpy arrays

to use a ridge penalty, the arg ridge_penalty is accessed at this point, during the .tran() function.

```
nnet.train(X_array, T_array, n_epochs_to_train, method = 'scg', ridge_penalty = 1.5)
```

To pass an array through a trained network and get outputs, use the .use() function and assign the return to a variable.
```
y = nnet.use(X_array)
```

## Authors

* **Charles Anderson** - *Neural Network Architect* - [Pattern Exploration](https://www.patternexploration.com)

* **David Anderson** - *Additional work* - [davidgraey](https://github.com/davidgraey)

* **www.patternexploration.com**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details