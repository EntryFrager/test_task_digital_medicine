import numpy as np

from Modules.LinearLayer import Linear
from Modules.SequentialContainer import Sequential
from Modules.BatchNormalization import BatchNormalization, ChannelwiseScaling
from Modules.LogSoftMaxFunction import LogSoftMax
from Modules.Optimizers import sgd_momentum, adam_optimizer
from Modules.ActivationFunctions import ReLU


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


def train(net, criterion, optimizer_name, n_epoch,
          X_train, y_train, X_val, y_val, batch_size):

    loss_train_history = []
    loss_val_history = []
    optimizer_config, optimizer_state = get_optimizer(optimizer_name)

    for i in range(n_epoch):
        print('Epoch {}/{}:'.format(i, n_epoch - 1), flush = True)

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
    net.evaluate()
    
    num_batches  = X_test.shape[0] / batch_size
    running_loss = 0.
    running_acc  = 0.

    for x_batch, y_batch in get_batches((X_test, y_test), batch_size):
        net.zeroGradParameters()

        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)

        running_loss += loss
        running_acc  += (predictions.argmax(axis = 1) == y_batch.argmax(axis = 1)).astype(float).mean()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_acc / num_batches
    return epoch_loss, epoch_acc


def get_net(activation = ReLU, norm = False):
    net = Sequential()

    net.add(Linear(28 * 28, 100))
    if norm:
        net.add(BatchNormalization(alpha = 0.001))
        net.add(ChannelwiseScaling(100))
    net.add(activation())

    net.add(Linear(100, 10))
    if norm:
        net.add(BatchNormalization(alpha = 0.001))
        net.add(ChannelwiseScaling(10))
    net.add(LogSoftMax())
    return net
