import numpy as np
import gzip

from sklearn.model_selection import ParameterSampler

from sklearn.model_selection import KFold

from Modules.SequentialContainer import Sequential
from Modules.Conv2d import Conv2d
from Modules.MaxPool2d import MaxPool2d
from Modules.FlattenLayer import Flatten
from Modules.LinearLayer import Linear
from Modules.LogSoftMaxFunction import LogSoftMax
from Modules.ActivationFunctions import ReLU
from Modules.Optimizers import sgd_momentum, adam_optimizer
from Modules.Criterions import ClassNLLCriterion


def load_data():
    X_train = load_image('data/train-images-idx3-ubyte.gz')
    X_test = load_image('data/t10k-images-idx3-ubyte.gz')
    Y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
    Y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    Y_train, Y_val = Y_train[:-10000], Y_train[-10000:]

    hot_y_train = one_hot_encode(Y_train)
    hot_y_val = one_hot_encode(Y_val)
    hot_y_test = one_hot_encode(Y_test)

    X_train = X_train.reshape(-1, 1, 28, 28)
    X_val   = X_val.reshape(-1, 1, 28, 28)
    X_test  = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_val, X_test, hot_y_train, hot_y_val, hot_y_test


def load_image(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        
    data = data.reshape(-1, 28, 28)
    
    return (data / np.float32(256)).squeeze()


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        
    return data


def one_hot_encode(y):
    return np.eye(10)[y]


def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        if start + batch_size <= n_samples:
            end = start + batch_size
        
            batch_idx = indices[start:end]
            yield X[batch_idx], Y[batch_idx]



def get_optimizer(optimizer_name):
    if optimizer_name == 'sgd_momentum':
        optimizer_config = {'learning_rate': 0.01,
                            'momentum': 0.9}
        optimizer_state = {}

    elif optimizer_name == 'adam_optimizer':
        optimizer_config = {'learning_rate': 0.001,
                            'beta1': 0.9,
                            'beta2': 0.999,
                            'epsilon': 1e-8}
        optimizer_state = {}

    else:
        raise NameError('Optimizer name have to one of {\'sgd_momentum\', \'adam_optimizer\'}')

    return optimizer_config, optimizer_state


def create_cnn(kernel_size, out_channels):
    CNN = Sequential()
    CNN.add(Conv2d(in_channels = 1, out_channels = out_channels, kernel_size = kernel_size))
    CNN.add(MaxPool2d(kernel_size = kernel_size))
    CNN.add(ReLU())
    CNN.add(Flatten())

    pad = (kernel_size - (28 % kernel_size)) % kernel_size
    in_features = out_channels * (((pad + 28) // kernel_size) ** 2)
    
    CNN.add(Linear(in_features, 10))
    CNN.add(LogSoftMax())
    return CNN


def train(net, criterion, optimizer_name, optimizer_config,
          n_epoch, X_train, y_train, X_val, y_val, batch_size):

    loss_train_history = []
    loss_val_history = []
    optimizer_state = {}

    for i in range(n_epoch):
        print('Epoch {}/{}:'.format(i + 1, n_epoch), flush = True)

        for phase in ['train', 'val']:
            if phase == 'train':
                X = X_train
                y = y_train
                net.train()
            else:
                X = X_val
                y = y_val
                net.evaluate()

            num_batches = X.shape[0] / batch_size
            running_loss = 0.
            running_acc = 0.

            for x_batch, y_batch in get_batches((X, y), batch_size):

                net.zeroGradParameters()

                predictions = net.forward(x_batch)
                loss = criterion.forward(predictions, y_batch)

                if phase == 'train':
                    gradOutput = criterion.backward(predictions, y_batch)
                    net.backward(x_batch, gradOutput)

                    variables = net.getParameters()
                    gradients = net.getGradParameters()

                    if optimizer_name == 'sgd_momentum':
                        sgd_momentum(variables, gradients, optimizer_config, optimizer_state)
                    else:
                        adam_optimizer(variables, gradients, optimizer_config, optimizer_state)

                running_loss += loss
                running_acc  += np.sum(predictions.argmax(axis = 1) == y_batch.argmax(axis = 1))
            
            epoch_loss = running_loss / num_batches
            epoch_acc  = running_acc  / y.shape[0]

            if phase == 'train':
                loss_train_history.append(epoch_loss)
            else:
                loss_val_history.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush = True)

    return net, loss_train_history, loss_val_history


def test(net, criterion, X_test, y_test, batch_size):
    X_test, y_test = X_test[:3200], y_test[:3200]
    net.evaluate()
    num_batches = X_test.shape[0] / batch_size
    running_loss = 0.
    running_acc = 0.
    for x_batch, y_batch in get_batches((X_test, y_test), batch_size):
        net.zeroGradParameters()

        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)
        running_loss += loss
        running_acc += (predictions.argmax(axis=1) == y_batch.argmax(axis=1)).astype(float).mean()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_acc / num_batches
    print('Final Test Loss: {:.4f} Final Test Acc: {:.4f}'.format(epoch_loss, epoch_acc), flush=True)

    return epoch_loss, epoch_acc


X_train, X_val, X_test, hot_y_train, hot_y_val, hot_y_test = load_data()

param_grid = {
    'kernel_size': [1, 3, 5, 7],
    'out_channels': [4, 8, 16, 32],
    'lr': [0.05, 0.1, 0.01, 0.01],
    'momentum': [0.8, 0.9, 0.99],
}

param_list = list(ParameterSampler(param_grid, n_iter = 3, random_state = 42))
print(param_list)
assert len(param_list) == 3

best_loss = np.inf
best_params = None
results = []
n_epoch = 3
batch_size = 32
criterion = ClassNLLCriterion()

for params in param_list:
    net = create_cnn(params['kernel_size'], params['out_channels'])
    optimizer_config = {'learning_rate': params['lr'], 'momentum': params['momentum']}

    net, loss_train_history, loss_val_history = train(
        net, criterion, 'sgd_momentum', optimizer_config,
        n_epoch, X_train, hot_y_train, X_val, hot_y_val, batch_size
    )

    final_val_loss = loss_val_history[-1]
    results.append((params, final_val_loss))

    if final_val_loss < best_loss:
        best_loss = final_val_loss
        best_params = params

print("Best hyperparameters:", best_params)
print("Best validation loss: {:.4f}".format(best_loss))

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

X_all = np.concatenate([X_train, X_val], axis = 0)
y_all = np.concatenate([hot_y_train, hot_y_val], axis = 0)

best_nn_weights = None
best_cv_loss = np.inf
cv_losses = []
n_epoch = 3
batch_size = 32
criterion = ClassNLLCriterion()

for fold, (train_index, val_index) in enumerate(kf.split(X_all)):
    X_train_cv, X_val_cv = X_all[train_index], X_all[val_index]
    y_train_cv, y_val_cv = y_all[train_index], y_all[val_index]

    net = create_cnn(best_params['kernel_size'], best_params['out_channels'])
    optimizer_config = {'learning_rate': best_params['lr'], 'momentum': best_params['momentum']}

    net, loss_train_history, loss_val_history = train(
        net, criterion, 'sgd_momentum', optimizer_config,
        n_epoch, X_train_cv, y_train_cv, X_val_cv, y_val_cv, batch_size
    )

    fold_val_loss = loss_val_history[-1]
    cv_losses.append(fold_val_loss)

    if fold_val_loss < best_cv_loss:
        best_cv_loss = fold_val_loss
        best_nn_weights = net.getParameters()

net = create_cnn(best_params['kernel_size'], best_params['out_channels'])
net.setParameters(best_nn_weights)

epoch_loss, epoch_acc = test(net, criterion, X_test, hot_y_test, batch_size = 32)