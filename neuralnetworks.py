######################################################################
# Written by Chuck Anderson
#   Colorado State University (chuck.anderson@colostate.edu)
#   Pattern Exploration LLC (chuck.anderson@patternexploration.com)
# Additional work by David Anderson
#   Pattern Exploration LLC (david.anderson@patternexploration.com)
######################################################################

import numpy as np
import torch
import optimizers as opt
import matplotlib.pyplot as plt
import copy


class NeuralNetwork:

    def __init__(self, n_inputs, n_hiddens_list, n_outputs):

        if not isinstance(n_hiddens_list, list):
            raise Exception('NeuralNetwork: n_hiddens_list must be a list.')
 
        if len(n_hiddens_list) == 0:
            self.n_hidden_layers = 0
        elif n_hiddens_list[0] == 0:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)
            
        self.n_inputs = n_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_outputs = n_outputs
        
        # Do we have any hidden layers?
        self.Vs = []
        ni = n_inputs
        # Initialize weights in hidden layers
        for layeri in range(self.n_hidden_layers):
            n_in_layer = self.n_hiddens_list[layeri]
            self.Vs.append(1 / np.sqrt(1 + ni) * np.random.uniform(-1, 1, size=(1 + ni, n_in_layer)))
            ni = n_in_layer
        # Initialize weights in output layer
        self.W = 1/np.sqrt(1 + ni) * np.random.uniform(-1, 1, size=(1 + ni, n_outputs))

        # Member variables for standardization
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.ridge_penalty = 0
        
        self.trained = False
        self.reason = None
        self.error_trace = None
        self.n_epochs = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.n_inputs}, {self.n_hiddens_list}, {self.n_outputs})'
        if self.trained:
            str += f'\n   Network was trained for {self.n_epochs} epochs'
            str += f' that took {self.training_time:.4f} seconds. Final objective value is {self.error_trace[-1]:.3f}'
        else:
            str += '  Network is not trained.'
        return str

    def _standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result

    def _unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans

    def _standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result

    def _unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans

    def _pack(self, Vs, W):
        return np.hstack([V.flat for V in Vs] + [W.flat])

    def _unpack(self, w):
        first = 0
        n_this_layer = self.n_inputs
        for i in range(self.n_hidden_layers):
            self.Vs[i][:] = w[first:first + (1 + n_this_layer) * 
                              self.n_hiddens_list[i]].reshape((1 + n_this_layer, self.n_hiddens_list[i]))
            first += (1 + n_this_layer) * self.n_hiddens_list[i]
            n_this_layer = self.n_hiddens_list[i]
        self.W[:] = w[first:].reshape((1 + n_this_layer, self.n_outputs))
    def _forward_pass(self, X):
        # Assume weights already unpacked
        Z_prev = X  # output of previous layer
        Z = [Z_prev]
        for i in range(self.n_hidden_layers):
            V = self.Vs[i]
            Z_prev = np.tanh(Z_prev @ V[1:, :] + V[0:1, :])
            Z.append(Z_prev)
        Y = Z_prev @ self.W[1:, :] + self.W[0:1, :]
        return Y, Z

    def _objectiveF(self, w, X, T):
        self._unpack(w)
        Y, _ = self._forward_pass(X)
        mean_square_error = 0.5 * np.mean((T - Y)**2)
        if self.ridge_penalty > 0:
            # Calculate ridge penalty to add to mean square error
            # Ridge penalty only calculated for original inputs
            # Weights in first layer
            W = self.W if self.n_hidden_layers == 0 else self.Vs[0]
            W = W[1:, :]
            self.ridge_penalty_count = W.size  # saved here to be used in _backwardPass in layers
            penalty = self.ridge_penalty * 0.5 * (W**2).sum() / W.size
        else:
            penalty = 0
        return mean_square_error + penalty

    def _gradientF(self, w, X, T):
        self._unpack(w)
        Y, Z = self._forward_pass(X)
        # Do backward pass, starting with delta in output layer
        delta = -(T - Y) / (X.shape[0] * T.shape[1])
        dW = np.vstack((np.sum(delta, axis=0), Z[-1].T @ delta))
        if self.n_hidden_layers == 0 and self.ridge_penalty > 0:
            W = self.W[1:, :]
            penalty_gradient = self.ridge_penalty * W / W.size
            dW[1:, :] += penalty_gradient
        dVs = []
        delta = (1 - Z[-1]**2) * (delta @ self.W[1:, :].T)
        for Zi in range(self.n_hidden_layers, 0, -1):
            Vi = Zi - 1  # because X is first element of Z
            dV = np.vstack((np.sum(delta, axis=0), Z[Zi-1].T @ delta))

            if Zi > 1:
                # not first layer yet, so calculate delta to
                # back-propagate to previous layer
                delta = (delta @ self.Vs[Vi][1:, :].T) * (1 - Z[Zi-1]**2)
            elif self.ridge_penalty > 0:
                # We are at first layer. Add ridge penalty.
                # penalty_gradient = self.ridge_penalty * self.
                ni = self.n_inputs
                W = self.Vs[0][1:, :]
                penalty_gradient = self.ridge_penalty * W / W.size
                dV[1:, :]  += penalty_gradient

            dVs.insert(0, dV)  # like append, but at front of list of d

        return self._pack(dVs, dW)

    def _setup_standardize(self, X, T):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy.copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy.copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1
        
    def _objective_to_actual(self, objective):
        return np.sqrt(objective)
    
    def train(self, X, T, n_epochs, method='scg',
              verbose=False, save_weights_history=False,
              learning_rate=0.001, momentum_rate=0.0,  # only for sgd and adam
              ridge_penalty=0):

        if X.shape[1] != self.n_inputs:
            raise Exception(f'train: number of columns in X ({X.shape[1]}) not equal to number of network inputs ({self.n_inputs})')
        
        self.ridge_penalty = ridge_penalty
        
        self._setup_standardize(X, T)
        X = self._standardizeX(X)
        T = self._standardizeT(T)
        
        try:
            algo = [opt.sgd, opt.adam, opt.scg][['sgd', 'adam', 'scg'].index(method)]
        except:
            raise Exception("train: method={method} not one of 'scg', 'sgd' or 'adam'")            

        result = algo(self._pack(self.Vs, self.W),
                      self._objectiveF,
                      [X, T], n_epochs,
                      self._gradientF,  # not used if scg
                      eval_f=self._objective_to_actual,
                      learning_rate=learning_rate, momentum_rate=momentum_rate,
                      verbose=verbose,
                      save_wtrace=save_weights_history)

        self._unpack(result['w'])
        self.reason = result['reason']
        self.error_trace = result['ftrace'] # * self.Tstds # to _unstandardize the MSEs
        self.n_epochs = len(self.error_trace) - 1
        self.trained = True
        self.weight_history = result['wtrace'] if save_weights_history else None
        self.training_time = result['time']
        return self

    def use(self, X, all_outputs=False):
        X = self._standardizeX(X)
        Y, Z = self._forward_pass(X)
        Y = self._unstandardizeT(Y)
        return (Y, Z[1:]) if all_outputs else Y

    def get_n_epochs(self):
        return self.n_epochs

    def get_error_trace(self):
        return self.error_trace

    def get_training_time(self):
        return self.training_time

    def get_weight_history(self):
        return self.weight_history

if __name__ == '__main__':

    X = 1 + np.arange(10).reshape((-1, 1))
    n = X.shape[0]
    T = X ** 2 * np.sin(X)
    n_epochs = 500
    X = np.hstack((X, X))  #  + np.random.uniform(-1, 1, (n, 1))))

    def rmse(Y, T):
        return np.sqrt(np.mean((T - Y)**2))
    
    nnet = NeuralNetwork(2, [], 1)
    # Equivalent to
    # nnet = NeuralNetwork(2, [0], 1)
    nnet.train(X, T, n_epochs)
    Y = nnet.use(X)
    print(f'scg  {nnet.n_hiddens_list} RMSE {rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')

    nnet = NeuralNetwork(2, [10, 5, 5, 5], 1)
    nnet.train(X, T, n_epochs, method='scg')
    Y = nnet.use(X)

    print(f'scg  {nnet.n_hiddens_list} RMSE {rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')

    nnet = NeuralNetwork(2, [10, 5, 5, 5], 1)
    nnet.train(X, T, n_epochs, method='sgd', learning_rate=0.5, momentum_rate=0.5)
    Y = nnet.use(X)
    print(f'sgd  {nnet.n_hiddens_list} RMSE {rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')
    
    nnet = NeuralNetwork(2, [10, 5, 5, 5], 1)
    nnet.train(X, T, n_epochs, method='adam', learning_rate=0.1)
    Y = nnet.use(X)
    print(f'adam {nnet.n_hiddens_list} RMSE {rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')

    # nnet = NeuralNetwork(2, [10, 5, 5, 5], 1)
    nnet = NeuralNetwork(2, [10], 1)
    # nnet.train(X, T, n_epochs, method='scg', ridge_penalty=10)
    nnet.train(X, T, 10000, method='scg', ridge_penalty=10)
    Y = nnet.use(X)
    print(f'scg with ridge penalty {nnet.n_hiddens_list} RMSE {rmse(Y, T):.3f} took {nnet.training_time:.3f} seconds')
