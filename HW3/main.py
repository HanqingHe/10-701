import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from HW3.nn import NN
from HW3.util import plot_confusion_matrix, mpl_default_setting, plot_image


CATEGORIES = {0: 'T-shirt',
              1: 'Trouser',
              2: 'Pullover',
              3: 'Dress',
              4: 'Coat',
              5: 'Sandal',
              6: 'Shirt',
              7: 'Sneaker',
              8: 'Bag',
              9: 'Ankle boot'}


def plot_loss(losses: dict, var_param: str, ylim: tuple = (-0.5, 10)):
    for param_val, loss in losses.items():
        plt.plot(loss['train_loss'], label=str(param_val))
    plt.title(f'Training loss vs {var_param.lower()}')
    plt.legend(title=var_param.capitalize())
    plt.ylim(ylim)
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.show()
    for batch_size, loss in losses.items():
        plt.plot(loss['test_loss'], label=str(batch_size))
    plt.title(f'Test loss vs {var_param.lower()}')
    plt.legend(title=var_param.capitalize())
    plt.ylim(ylim)
    plt.xlabel('Epoch')
    plt.ylabel('Test loss')
    plt.show()


if __name__ == '__main__':
    dset_dirpath = './data/'
    train_filename = 'train.csv'
    test_filename = 'test.csv'
    init_param_dir = './params'

    train_path = os.path.join(dset_dirpath, train_filename)
    test_path = os.path.join(dset_dirpath, test_filename)

    # Load dataset (pd.read_csv is faster than np.genfromtxt)
    train = pd.read_csv(train_path, header=None).to_numpy()
    test = pd.read_csv(test_path, header=None).to_numpy()
    # Split labels from features
    x_slice = np.s_[:, :-1]  # (n_samples, n_pixels)
    y_slice = np.s_[:, -1]  # (n_samples,)
    train_x, train_y = train[x_slice], train[y_slice]
    test_x, test_y = test[x_slice], test[y_slice]

    mpl_default_setting()

    # Q6.1 ~ Q6.3
    nn = NN(n_hidden_nodes=256,
            method='load')
    xy_slice = np.s_[:1]  # slice the first data point
    nn.fit(train_x[xy_slice], train_y[xy_slice],
           test_x[xy_slice], test_y[xy_slice],
           batch_size=1, epoch=1, lr=0.01, store_vars=True)
    int_vars = nn.int_vars
    print(f"Q6.1 -- a10 = {int_vars['o1'].flatten()[9]:.4f}")
    print(f"Q6.2 -- z20 = {int_vars['a1'].flatten()[19]:.4f}")
    print(f"Q6.3 -- class = {int(np.argmax(int_vars['a2']))}")

    # Q6.4
    nn = NN(n_hidden_nodes=256,
            method='load')
    nn.fit(train_x, train_y,
           test_x, test_y,
           batch_size=1, epoch=3, lr=0.01)
    print(f"Q6.4 --")
    for k, b2_k in enumerate(nn.biases[1]):
        print(f"beta(k,0) = {b2_k:.4f}")

    # Q6.5 ~ Q6.6
    nn = NN(n_hidden_nodes=256,
            method='load')
    losses = nn.fit(train_x, train_y,
                    test_x, test_y,
                    batch_size=1, epoch=15, lr=0.01)
    print(f"Q6.5 ~ Q6.6 --")
    for i_epoch, (loss, acc) in enumerate(zip(losses['test_loss'], losses['test_acc'])):
        print(f"Epoch {i_epoch}: test loss = {loss:.4f}, test acc = {acc:.4f}")

    # Q6.7 ~ Q6.9
    # Q6.7
    nn = NN(n_hidden_nodes=256,
            method='load')
    losses = nn.fit(train_x, train_y,
                    test_x, test_y,
                    batch_size=1, epoch=100, lr=0.01)
    pickle.dump(nn, open('./models/Q6.7~Q6.9.pickle', 'wb'))
    print(f"Q6.7 -- final train loss = {losses['train_loss'][-1]:.4f}, final test acc = {losses['test_acc'][-1]:.4f}")
    # Q6.8
    # Load saved model
    nn = pickle.load(open('./models/Q6.7~Q6.9.pickle', 'rb'))
    train_y_pred, _ = nn.predict(train_x)
    test_y_pred, _ = nn.predict(test_x)
    cm_train = confusion_matrix(train_y, train_y_pred)
    cm_test = confusion_matrix(test_y, test_y_pred)
    plot_confusion_matrix(cm_train, classes=np.arange(10), title='Confusion matrix (training set)')
    plot_confusion_matrix(cm_test, classes=np.arange(10), title='Confusion matrix (test set)')
    # Q6.9
    miss = test_y != test_y_pred
    for k in (0, 5, 6, 9):
        mask = np.logical_and(miss, test_y == k)
        err_sample = test_x[mask][0]
        err_label = CATEGORIES[test_y_pred[mask][0]]
        plot_image(err_sample, title=f'{CATEGORIES[k]} (misclassified as {err_label})')

    # Q6.10
    models = {}
    losses = {}
    for batch_size in (1, 10, 50, 100):
        nn = NN(n_hidden_nodes=256,
                method='load')
        losses[batch_size] = nn.fit(train_x, train_y,
                                    test_x, test_y,
                                    batch_size=batch_size, epoch=100, lr=0.01)
        models[batch_size] = nn
    pickle.dump({'models': models, 'losses': losses}, open('./models/Q6.10.pickle', 'wb'))
    res: dict = pickle.load(open('./models/Q6.10.pickle', 'rb'))
    models, losses = res.values()
    plot_loss(losses, 'batch size', ylim=(-0.5, 10))

    # Q6.13
    # Learning rate
    models = {}
    losses = {}
    for lr in (0.001, 0.01, 0.1, 1):
        nn = NN(n_hidden_nodes=256,
                method='load')
        losses[lr] = nn.fit(train_x, train_y,
                            test_x, test_y,
                            batch_size=1, epoch=100, lr=lr)
        models[lr] = nn
    pickle.dump({'models': models, 'losses': losses}, open('./models/Q6.13_lr.pickle', 'wb'))
    res: dict = pickle.load(open('./models/Q6.13_lr.pickle', 'rb'))
    models, losses = res.values()
    plot_loss(losses, 'learning rate', ylim=(-0.5, 100))
    # Width of hidden layer
    models = {}
    losses = {}
    for n_hidden_nodes in (64, 128, 256, 512):
        nn = NN(n_hidden_nodes=n_hidden_nodes,
                method='xavier', shapes=(784, n_hidden_nodes, 10))
        losses[n_hidden_nodes] = nn.fit(train_x, train_y,
                                        test_x, test_y,
                                        batch_size=1, epoch=100, lr=0.01)
        models[n_hidden_nodes] = nn
    pickle.dump({'models': models, 'losses': losses}, open('./models/Q6.13_nodes.pickle', 'wb'))
    res: dict = pickle.load(open('./models/Q6.13_nodes.pickle', 'rb'))
    models, losses = res.values()
    plot_loss(losses, '# hidden nodes', ylim=(0.0, 1.5))
    # Initialization method
    models = {}
    losses = {}
    for method in ('zero', 'uniform', 'xavier', 'normal'):
        nn = NN(n_hidden_nodes=256,
                method=method, shapes=(784, 256, 10))
        losses[method] = nn.fit(train_x, train_y,
                                test_x, test_y,
                                batch_size=1, epoch=100, lr=0.01)
        models[method] = nn
    pickle.dump({'models': models, 'losses': losses}, open('./models/Q6.13_method.pickle', 'wb'))
    res: dict = pickle.load(open('./models/Q6.13_method.pickle', 'rb'))
    models, losses = res.values()
    plot_loss(losses, 'method', ylim=(-0.5, 10))

