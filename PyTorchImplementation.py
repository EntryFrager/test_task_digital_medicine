import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

#------------------------------------------------Network architecture-------------------------------------------------

class TorchNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 28 * 28, out_features = 512),
            nn.BatchNorm1d(num_features =  512, momentum = 0.1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features = 512, out_features = 256),
            nn.BatchNorm1d(num_features = 256, momentum = 0.1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features = 256, out_features = num_classes),
            nn.LogSoftmax(dim = 1)
        )

    
    def forward(self, x):
        return self.net(x)


def train(net, train_loader, val_loader, 
          n_epoch, optimizer, scheduler, criterion):
    loss_train_history = []
    loss_val_history = []

    for epoch in range(n_epoch):
        print('Epoch {}/{}:'.format(epoch + 1, n_epoch), flush = True)

        train_loss = train_acc = val_acc = val_loss = 0.0

        net.train()

        for (batch_idx, train_batch) in enumerate(train_loader):
            images, labels = train_batch[0].to(device), train_batch[1].to(device)
            optimizer.zero_grad()

            preds = net(images)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += (preds.argmax(dim = 1) == labels).sum().item()

        train_acc  /= len(train_loader.dataset)
        train_loss /= len(train_loader)
        loss_train_history.append(train_loss)

        net.eval()

        with torch.no_grad():
            for val_batch in val_loader:
                images, labels = val_batch[0].to(device), val_batch[1].to(device)
                preds = net(images)
                val_acc  += (preds.argmax(axis = 1) == labels).sum().item()
                val_loss += criterion(preds, labels).item()

        val_acc  /= len(val_loader.dataset)
        val_loss /= len(val_loader)
        loss_val_history.append(val_loss)

        scheduler.step(val_acc)

        print(f'train Loss: {train_loss:.4f} Acc: {train_acc:.4f}\n'
              f'val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    return net, loss_train_history, loss_val_history


def test(net, test_loader, criterion):
    net.eval()

    test_acc = test_loss = 0.0

    with torch.no_grad():
        for (batch_idx, test_batch) in enumerate(test_loader): 
            images, labels = test_batch[0].to(device), test_batch[1].to(device)
            preds = net(images)

            test_acc  += (preds.argmax(axis = 1) == labels).sum().item()
            test_loss += criterion(preds, labels).item()

    test_acc  /= len(test_loader.dataset)
    test_loss /= len(test_loader)

    return test_loss, test_acc


#-----------------------------------------------Set of hyperparameters------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size    = 64
learning_rate = 0.001
n_epoch       = 15
num_classes   = 10

criterion   = nn.NLLLoss()

#-----------------------------------------------Function Transform Image----------------------------------------------

transform_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])

#----------------------------------------------------Train Dataset----------------------------------------------------

train_dataset = torchvision.datasets.MNIST('./datasets/', train = True, download = True, transform = transform_image)

train_subset, val_subset = torch.utils.data.random_split(train_dataset, [50000, 10000], generator = torch.Generator().manual_seed(1))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size = batch_size, shuffle = False)

#----------------------------------------------------Test Dataset-----------------------------------------------------

test_dataset = torchvision.datasets.MNIST('./datasets/', train = False, download = True, transform = transform_image)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

#----------------------------------------------------Create network---------------------------------------------------

net = TorchNet(num_classes).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 2)

net, train_loss, val_loss = train(net, train_loader, val_loader, n_epoch, optimizer, scheduler, criterion)

test_loss, test_acc = test(net, test_loader, criterion)
print(f'\nFinal Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}')

plt.plot(train_loss, label = 'Train Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()