import os
from typing import Union, Tuple

import numpy as np


class NN:
    def __init__(self, n_hidden_nodes: int = 256,
                 init: Union[str, dict] = 'load',
                 init_param_dir: str = './params/'):
        r"""Neural network with a single hidden layer

        Args:
            n_hidden_nodes: int
            init: Union[str, dict]
                'load': load initial parameters from file (./params/)
                dict: {'weights': List[np.ndarray]
                                  Shapes: [(dim_input, n_hidden_nodes), (n_hidden_nodes, n_classes)]
                       'biases': List[np.ndarray]
                                 Shapes: [(n_hidden_nodes,), (n_classes,)]
            init_param_dir: str
                Directory storing parameters
        """
        if init == 'load':
            alpha1_path = os.path.join(init_param_dir, 'alpha1.txt')
            beta1_path = os.path.join(init_param_dir, 'beta1.txt')
            alpha2_path = os.path.join(init_param_dir, 'alpha2.txt')
            beta2_path = os.path.join(init_param_dir, 'beta2.txt')
            # Load parameters from file
            alpha1 = np.genfromtxt(alpha1_path, delimiter=',')
            beta1 = np.genfromtxt(beta1_path, delimiter=',')
            alpha2 = np.genfromtxt(alpha2_path, delimiter=',')
            beta2 = np.genfromtxt(beta2_path, delimiter=',')
            # Sanity check
            try:
                assert alpha1.shape[0] == beta1.shape[0] == alpha2.shape[1] == n_hidden_nodes
                assert alpha2.shape[0] == beta2.shape[0]
            except (AssertionError, IndexError):
                raise ValueError("Dimensions of parameters do not match")
            # Infer n_hidden_nodes
            self.n_hidden_nodes = len(beta1)
            # Store weights and biases
            self.weights = [alpha1, alpha2]
            self.biases = [beta1, beta2]
            # Store intermediate variables
            self.int_vars = {}
        elif isinstance(init, dict):
            try:
                assert len(init['weights']) == len(init['biases']) == 2
                alpha1, alpha2 = init['weights']
                beta1, beta2 = init['biases']
            except (AssertionError, KeyError):
                raise ValueError("Must input 2 weight matrices and 2 bias vectors")
            try:
                assert alpha1.shape[0] == beta1.shape[0] == alpha2.shape[1] == n_hidden_nodes
                assert alpha2.shape[0] == beta2.shape[0]
            except (AssertionError, IndexError):
                raise ValueError("Dimensions of parameters do not match")
            self.n_hidden_nodes = n_hidden_nodes
            self.weights = init['weights']
            self.biases = init['biases']

    def fit(self, train_x: np.ndarray, train_y: np.ndarray,
            test_x: np.ndarray, test_y: np.ndarray,
            batch_size: int = 1, epoch: int = 15, lr: float = 0.01,
            store_vars: bool = False) -> dict:
        try:
            assert len(train_x.shape) == len(test_x.shape) == 2
            assert len(train_y.shape) == len(test_y.shape) == 1
            assert train_x.shape[0] == train_y.shape[0]
            assert train_x.shape[1] == test_x.shape[1]
            assert test_x.shape[0] == test_y.shape[0]
        except AssertionError:
            raise ValueError("The shape of train_x should be (n_samples_train, n_features), "
                             "the shape of train_y should be (n_samples_train,), "
                             "the shape of test_x should be (n_samples_test, n_features), "
                             "the shape of test_y should be (n_samples_test,)")
        # One-hot encode train_y into a matrix
        _train_x = train_x
        _train_y = self.onehot_encoder(train_y)
        # Model training
        # Note: Notations
        #       i_epoch:         index of current epoch
        #       i_batch, xi, yi: index, features and labels of current batch
        #       w1, b1, o1, a1:  weights, biases, output (not activated) and output (activated)
        #                        of the 1st (hidden) layer
        #       w2, b2, o2, a2:  weights, biases, output (not activated) and output (activated)
        #                        of the 2nd (output) layer
        #       loss:            average cross entropy loss over the training batch
        #       dloss_dxx:       partial derivative of loss with respect to xx
        # Load weights and biases
        w1, w2 = self.weights
        b1, b2 = self.biases
        losses = {'avg_train_loss': [],
                  'test_loss': [],
                  'test_acc': []}
        for i_epoch in range(epoch):
            for i_batch, (xi, yi) in enumerate(self.iterate_batches(_train_x, _train_y, batch_size)):
                # Forward pass
                o1 = self.linear_forward(xi, w1, b1)
                a1 = self.sigmoid_forward(o1)
                o2 = self.linear_forward(a1, w2, b2)
                loss, a2 = self.softmax_xeloss_forward(o2, yi)
                # Backpropagation
                # Note: Notations
                #       n: index of samples in each minibatch (e.g. {1...50})
                #       i: index of features in the input (e.g. {1...784})
                #       j: index of features in the output of the hidden layer (e.g. {1...256})
                #       k: index of classes (e.g. {1...10})
                dloss_do2 = self.softmax_xeloss_backward(a2, yi)
                dloss_da1 = self.linear_backward(w2, dloss_do2)
                dloss_do1 = self.sigmoid_backward(a1, dloss_da1)
                dloss_dw2 = np.einsum('nk,nj->kj', dloss_do2, a1) / batch_size
                dloss_dw1 = np.einsum('nj,ni->ji', dloss_do1, xi) / batch_size
                dloss_db2 = np.einsum('nk->k', dloss_do2) / batch_size
                dloss_db1 = np.einsum('nj->j', dloss_do1) / batch_size
                # Gradient descent
                w1 -= lr * dloss_dw1
                w2 -= lr * dloss_dw2
                b1 -= lr * dloss_db1
                b2 -= lr * dloss_db2
                # Store intermediate variables when required
                if store_vars:
                    self.int_vars = {'a1': a1,
                                     'a2': a2,
                                     'b1': b1,
                                     'b2': b2,
                                     'o1': o1,
                                     'o2': o2}
            # Calculate losses
            avg_train_loss, _ = self.loss_and_acc(train_x, train_y)
            test_loss, test_acc = self.loss_and_acc(test_x, test_y)
            losses['avg_train_loss'].append(avg_train_loss)
            losses['test_loss'].append(test_loss)
            losses['test_acc'].append(test_acc)
        return losses

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Note: Notations
        #       w1, b1, o1, a1:  weights, biases, output (not activated) and output (activated)
        #                        of the 1st (hidden) layer
        #       w2, b2, o2, a2:  weights, biases, output (not activated) and output (activated)
        #                        of the 2nd (output) layer
        #       loss:            average cross entropy loss over the training batch
        # Forward pass
        w1, w2 = self.weights
        b1, b2 = self.biases
        o1 = self.linear_forward(x, w1, b1)
        a1 = self.sigmoid_forward(o1)
        o2 = self.linear_forward(a1, w2, b2)
        a2 = self.softmax(o2)
        y_pred = np.argmax(a2, axis=1)
        return y_pred, a2

    def loss_and_acc(self, x: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        y_pred, a2 = self.predict(x)
        xeloss = self.xeloss(a2, self.onehot_encoder(labels))
        acc = self.acc(y_pred, labels)
        return xeloss, acc

    @staticmethod
    def linear_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        try:
            assert weight.shape[0] == bias.shape[0]
            assert x.shape[1] == weight.shape[1]
        except (AssertionError, IndexError):
            raise ValueError("The shape of x should be (n_samples, n_features), "
                             "the shape of weight should be (dim_output, n_features), "
                             "the shape of bias should be (dim_output,)")
        out = np.einsum('ji,ni->nj', weight, x) + bias  # n: samples, i: input features, j: output features
        return out

    @staticmethod
    def linear_backward(weight: np.ndarray, dloss_dout: np.ndarray) -> np.ndarray:
        try:
            assert weight.shape[0] == dloss_dout.shape[1]
        except (AssertionError, IndexError):
            raise ValueError("The shape of weight should be (dim_output, n_features), "
                             "the shape of dloss_dout should be (n_samples, dim_output)")
        dout_dx = weight
        dloss_dx = np.einsum('nj,ji->ni', dloss_dout, dout_dx)  # n: samples, i: input features, j: output features
        return dloss_dx

    @staticmethod
    def sigmoid_forward(x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-x))
        return out

    @staticmethod
    def sigmoid_backward(out: np.ndarray, dloss_dout: np.ndarray) -> np.ndarray:
        try:
            assert len(out.shape) == len(dloss_dout.shape) == 2
            assert out.shape == dloss_dout.shape
        except AssertionError:
            raise ValueError("The shape of out and dloss_dout should be identical")
        dloss_dx = np.einsum('nj,nj,nj->nj', dloss_dout, out, 1 - out)  # n: samples, j: input/output features
        return dloss_dx

    def softmax_xeloss_forward(self, x: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
        r"""

        Args:
            x: np.ndarray, (n_samples, n_classes)
            labels: np.ndarray, (n_samples, n_classes)
                One-hot encoded class label matrix
        Returns:
            xeloss: float
            y_pred: np.ndarray, (n_samples, n_classes)
                One-hot encoded model prediction
        """
        try:
            assert len(x.shape) == len(labels.shape) == 2
            assert x.shape[0] == labels.shape[0]
        except AssertionError:
            raise ValueError("The shape of x and labels do not match")
        y_pred = self.softmax(x)
        xeloss = self.xeloss(y_pred, labels)
        return xeloss, y_pred

    @staticmethod
    def softmax_xeloss_backward(y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
        try:
            assert len(y_pred.shape) == len(labels.shape) == 2
            assert y_pred.shape == labels.shape
        except AssertionError:
            raise ValueError("The shape of y_pred and labels should be identical")
        return y_pred - labels

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        if len(x.shape) != 2:
            raise ValueError("x should be 2-dimensional")
        ex = np.exp(x)
        out = ex / np.einsum('nk->n', ex)[:, None]  # n: samples, k: classes
        return out

    @staticmethod
    def xeloss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        try:
            assert len(y_pred.shape) == len(y_true.shape) == 2
            assert y_pred.shape == y_true.shape
        except AssertionError:
            raise ValueError("y_pred and y_true should be 2-dimensional with the same shape")
        N = y_true.shape[0]  # number of samples
        xeloss = -np.einsum('nk,nk->', y_true, np.log(y_pred)) / N  # n: samples, k: classes
        return xeloss

    @staticmethod
    def acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        try:
            assert len(y_pred.shape) == len(y_true.shape) == 1
            assert y_pred.shape == y_true.shape
        except AssertionError:
            raise ValueError("y_pred and y_true should be 1-dimensional with the same shape")
        return np.average(y_pred.astype('int') == y_true.astype('int'))

    def onehot_encoder(self, y: np.ndarray) -> np.ndarray:
        y_encoded = np.zeros((y.shape[0], self.biases[1].shape[0]))
        y_encoded[np.arange(y.shape[0]), y.astype('int')] = 1  # https://stackoverflow.com/a/29831596/10690911
        return y_encoded

    @staticmethod
    def iterate_batches(x: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        # Sanity check
        assert isinstance(batch_size, int)
        assert len(x) == len(y)
        n_data = len(x)
        # Create iterator
        for idx_start in range(0, n_data - batch_size + 1, batch_size):
            batch_slice = np.s_[idx_start: idx_start + batch_size]
            yield x[batch_slice], y[batch_slice]
