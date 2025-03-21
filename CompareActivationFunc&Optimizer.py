import numpy as np
import gzip
import matplotlib.pyplot as plt

from ActivationFunctions import ReLU, ELU, LeakyReLU, SoftPlus
from Criterions import ClassNLLCriterion
from ExampleCnn import train, test, get_net


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


def one_hot_encode_test(hot_y_train):
    first_ten_answers = np.array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                  [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                  [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
    np.testing.assert_equal(hot_y_train[:10], first_ten_answers, err_msg="First ten samples are not equal")
    print("The test pass successfully !!!")


def dimension_test(X_train, X_val, X_test):
    true_train_shape = (50000, 1, 784)
    true_test_shape = (10000, 1, 784)
    np.testing.assert_equal(X_train.shape, true_train_shape, err_msg="Train shape doesn't the same")
    np.testing.assert_equal(X_val.shape, true_test_shape, err_msg="Validation shape doesn't the same")
    np.testing.assert_equal(X_test.shape, true_test_shape, err_msg="Test shape doesn't the same")
    print("The test pass successfully !!!")


X_train = load_image('data/train-images-idx3-ubyte.gz')
X_test = load_image('data/t10k-images-idx3-ubyte.gz')
Y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
Y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')

X_train, X_val = X_train[:-10000], X_train[-10000:]
Y_train, Y_val = Y_train[:-10000], Y_train[-10000:]

hot_y_train = one_hot_encode(Y_train)
hot_y_val = one_hot_encode(Y_val)
hot_y_test = one_hot_encode(Y_test)

# one_hot_encode_test(hot_y_train)

X_train = X_train.reshape(28, 28, 1, -1)
X_val   = X_val.reshape(28, 28, 1, -1)
X_test  = X_test.reshape(28, 28, 1, -1)

# dimension_test(X_train, X_val, X_test)

batch_size = 64
n_epoch = 15
criterion = ClassNLLCriterion()
optimizer_name = 'sgd_momentum'

nets = []
activations = [ReLU, LeakyReLU, ELU, SoftPlus]

for activ in activations:
    nets.append(get_net(activ, norm = False))
    nets.append(get_net(activ, norm = True))

losses_train = []
losses_val   = []

for i, net in enumerate(nets):
    print(f'\n\nTrain {i}/{len(nets)}')
    net, train_loss, val_loss = train(net, criterion, optimizer_name, n_epoch, 
                                      X_train, hot_y_train, X_val, hot_y_val, batch_size)
    losses_train.append(train_loss)
    losses_val.append(val_loss)

for net in nets:
    test_loss, test_acc = test(net, criterion, X_test, Y_test, batch_size)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

for i in range(len(nets)):
    plt.plot(losses_train[i], label = f'Net {i + 1} train')
    plt.plot(losses_val[i], label = f'Net {i + 1} val')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Compare activation functions")
plt.show()

nets = []
optimizer_names = ['sgd_momentum', 'adam_optimizer']

for optim_name in optimizer_names:
    nets.append(get_net(activation = ReLU, norm = True))

criterion = ClassNLLCriterion()
batch_size = 64
n_epoch = 15

losses_train = []
losses_val   = []

for i, (net, optim_name) in enumerate(nets, optimizer_names):
    print(f'\n\nTrain {i}/{len(nets)}')
    net, train_loss, val_loss = train(net, criterion, optim_name, n_epoch, 
                                      X_train, hot_y_train, X_val, hot_y_val, batch_size)
    losses_train.append(train_loss)
    losses_val.append(val_loss)

for net in nets:
    test_loss, test_acc = test(net, criterion, X_test, Y_test, batch_size)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

for i in range(len(nets)):
    plt.plot(losses_train[i], label = f'Net {i + 1} train')
    plt.plot(losses_val[i], label = f'Net {i + 1} val')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Compare optimizers")
plt.show()