import numpy as np
import gzip
import matplotlib.pyplot as plt

from Modules.ActivationFunctions import ReLU, ELU, LeakyReLU, SoftPlus
from Modules.Criterions import ClassNLLCriterion
from ExampleCnn import train, test, get_net


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

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val   = X_val.reshape(X_val.shape[0], -1)
    X_test  = X_test.reshape(X_test.shape[0], -1)

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

#-----------------------------------------------------Load dataset and preparing data----------------------------------------------------

X_train, X_val, X_test, hot_y_train, hot_y_val, hot_y_test = load_data()

#------------------------------------------------------Compare activation functions------------------------------------------------------

batch_size = 64
n_epoch = 15
criterion = ClassNLLCriterion()
optimizer_name = 'sgd_momentum'

nets = []
activations = [ReLU, LeakyReLU, ELU, SoftPlus]      # LeakyReLU выдала лучшие результаты для тестовых данных, LeakyReLU + BN выдала лучшие 

for activ in activations:
    nets.append(get_net(activ, norm = False))
    nets.append(get_net(activ, norm = True))

losses_train = []
losses_val   = []

for i, net in enumerate(nets):
    print(f'\n\nTrain {i + 1}/{len(nets)}')
    net, train_loss, val_loss = train(net, criterion, optimizer_name, n_epoch, 
                                      X_train, hot_y_train, X_val, hot_y_val, batch_size)
    losses_train.append(train_loss)
    losses_val.append(val_loss)
    
for net in nets:
    test_loss, test_acc = test(net, criterion, X_test, hot_y_test, batch_size)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

plt.figure(figsize = (14, 10))

for i, (lt, lv) in enumerate(zip(losses_train, losses_val)):
    label = f'{activations[i // 2].__name__} {"+BN" if i % 2 else ""}'
    plt.plot(lt, label = f'Train {label}')
    plt.plot(lv, '--', label = f'Val {label}')

print("Losses_train:", losses_train)
print("Losses_val:", losses_val)

plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Compare activation function')
plt.show()

#-----------------------------------------------------------Compare optimizers-----------------------------------------------------------

nets = []
optimizer_names = ['sgd_momentum', 'adam_optimizer']

for optim_name in optimizer_names:
    nets.append(get_net(activation = ReLU, norm = True))

losses_train = []
losses_val = []

for i, net in enumerate(nets):
    print(f'\n\nTrain {i + 1}/{len(nets)}')
    net, train_loss, val_loss = train(net, criterion, optimizer_name, n_epoch, 
                                      X_train, hot_y_train, X_val, hot_y_val, batch_size)
    losses_train.append(train_loss)
    losses_val.append(val_loss)

for net in nets:
    test_loss, test_acc = test(net, criterion, X_test, hot_y_test, batch_size)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

plt.figure(figsize = (14, 10))

for i, (lt, lv) in enumerate(zip(losses_train, losses_val)):
    label = f'{optimizer_names[i]}'
    plt.plot(lt, label = f'Train {label}')
    plt.plot(lv, '--', label = f'Val {label}')

print("Losses_train:", losses_train)
print("Losses_val:", losses_val)

plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Compare optimizers')
plt.show()