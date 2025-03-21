import numpy as np
import gzip
import matplotlib.pyplot as plt

from ActivationFunctions import ReLU
from Criterions import ClassNLLCriterion
from ExampleCnn import test
from LinearLayer import Linear
from SequentialContainer import Sequential
from BatchNormalization import BatchNormalization, ChannelwiseScaling
from LogSoftMaxFunction import LogSoftMax
from DropoutFunction import Dropout
from Optimizers import sgd_momentum, adam_optimizer


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
    one_hot_y = np.zeros((y.shape[0], 10))
    one_hot_y[np.arange(y.shape[0]), y] = 1

    # one_hot_y = np.eye(10)[y]

    return one_hot_y


def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]


def get_optimizer(optimizer_name):
    if optimizer_name == 'sgd_momentum':
        optimizer_config = {'learning rate': 0.01,
                            'momentum': 0.9}
        optimizer_state = {}

    elif optimizer_name == 'adam_optimizer':
        optimizer_config = {'learning rate': 0.01,
                            'beta1': 0.9,
                            'beta2': 0.999,
                            'epsilon': 1e-8}
        optimizer_state = {}

    else:
        raise NameError('Optimizer name have to one of {\'sgd_momentum\', \'adam_optimizer\'}')

    return optimizer_config, optimizer_state


def augment_batch(images):
    augmented = []
    for img in images:
        img = img.reshape(28, 28)
    
    return augmented


def get_improved_net():
    net = Sequential()
    
    net.add(Linear(28*28, 512))
    net.add(BatchNormalization(alpha=0.9))
    net.add(ChannelwiseScaling(512))
    net.add(ReLU())
    net.add(Dropout(0.3))
    
    net.add(Linear(512, 256))
    net.add(BatchNormalization(alpha=0.9))
    net.add(ChannelwiseScaling(256))
    net.add(ReLU())
    net.add(Dropout(0.3))
    
    net.add(Linear(256, 10))
    net.add(LogSoftMax())
    
    return net


def train_with_augmentation(net, criterion, optimizer_name, n_epoch,
                           X_train, y_train, X_val, y_val, batch_size):
    loss_train_history = []
    loss_val_history = []
    optimizer_config, optimizer_state = get_optimizer(optimizer_name)
    
    for epoch in range(n_epoch):
        print('Epoch {}/{}:'.format(epoch, n_epoch - 1), flush = True)
         
        for phase in ['train', 'val']:
            if phase == 'train':
                X = X_train
                y = y_train
                net.train()
            else:
                X = X_val
                y = y_val
                net.evaluate()
            
            running_loss = 0.0
            running_acc = 0.0
            num_batches = X.shape[0] / batch_size
            
            for x_batch, y_batch in get_batches((X, y), batch_size):
                # if phase == 'train':
                #     x_batch = augment_batch(x_batch)
                
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
                running_acc += np.sum(predictions.argmax(axis=1) == y_batch.argmax(axis=1))
            
            epoch_loss = running_loss / num_batches
            epoch_acc  = running_acc  / y.shape[0]
            
            if phase == 'train':
                loss_train_history.append(epoch_loss)
            else:
                loss_val_history.append(epoch_loss)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return net, loss_train_history, loss_val_history


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
    return X_train, X_val, X_test, hot_y_train, hot_y_val, hot_y_test


def main():
    X_train, X_val, X_test, hot_y_train, hot_y_val, hot_y_test = load_data()
    
    batch_size = 128
    n_epoch = 30
    criterion = ClassNLLCriterion()
    
    net = get_improved_net()
    
    net, train_loss, val_loss = train_with_augmentation(net, criterion, 'sgd_momentum', n_epoch,
                                                        X_train, hot_y_train, X_val, hot_y_val, batch_size)
    
    test_loss, test_acc = test(net, criterion, X_test, hot_y_test, batch_size)
    print(f'\nFinal Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}')
    
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()