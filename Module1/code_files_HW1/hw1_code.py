import hw1
import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('default')
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot
import seaborn as sns
##Prob 1.2.a
# form the train_dataset 
# compute SGD 
def regr(X, Y, optimizer, model, criterion, n_iter = 50, batch_size = 10, eta = 0.1):
    return logistic_regr(X, Y, n_iter, batch_size, eta, optimizer, model, criterion)

class LogisticRegression(torch.nn.Module):
  def __init__(self):
    super(LogisticRegression, self).__init__()
    self.linear = torch.nn.Linear(2, 1, bias=False)

  def forward(self, x):
    y_hat = self.linear(x)
    return torch.sigmoid(y_hat)


def logistic_regr( X, Y, n_iter, batch_size, eta, optimizer, model, criterion):
    n_samples, n_features = X.shape
    # convert numpy arrays to torch arrays
    Xt = torch.tensor(X, dtype=torch.float)
    Yt = torch.tensor(Y, dtype=torch.float).reshape(-1, 1)
    history = []
    # epochs -- terminology as running through all the data set once is one epoch.
    for i in range(n_iter):
        # each epoch we set different permutation [3, 0, 1, 2] ---> next [2, 3, 1, 0]
        permutation = torch.randperm(n_samples) # create randomly numbers from [0, n_samples], e.g [3, 0, 1, 2]
        # loss is the thing we try to minimize. 
        epoch_loss = 0
        # batch_size is how many data sets we used for eazh gradient update.
        for batch_start in range(0, n_samples, batch_size):
            # index is the choosn index e.g [3] that we choose from X\Y. And in the next round of this loop, it will be [0]
            idx = permutation[batch_start:batch_start + batch_size]
            # X\Ybatch are the X\Y that we used in this batch.
            Xbatch = Xt[idx, :]
            Ybatch = Yt[idx]
            # minimize cross entropy
            loss = criterion
            #construct target = actual Y
            target = Ybatch
            #get output
            loss_batch = loss(model(Xbatch), target)
            # gradience will be computed here in background. We don't see it but it will update in optimizer.
            # .item() converts a 1-element tensor to a Python float
            epoch_loss += loss_batch.item()
            # reset gradients from previous batch
            optimizer.zero_grad()
            # compute the gradients for this batch
            loss_batch.backward()
            # the acuall gradients computation happens. Set gradient which comes from loss_batch -- its computation. 
            # update the parameters
            optimizer.step()              
        history.append(epoch_loss)
    return history

##Prob 1.2.a compare with the learn
# compared with the SKLearn
# compare BCE and SKL accuracy
def compare_classification(a, b, data):
    print(data,' dataset')
    if a > b:
        print('SKL',a, ' is better than BCE',b)
    elif a < b:
        print('BCE',b, ' is better than SKL',a)
    else:
        print('SKL',a, ' is the same as BCE',b)
    return;
    

#collect the Xtest data which Ypred == 1
def collect(x, yd, y):
    df = pd.DataFrame( data = x, columns = ['x0', 'x1']  )
    df['Ypred'] = yd
    df['Y'] = y
    x0 = df[ (df['Ypred'] == 0 )& (df['Y'] == 0 )]
    x1 = df[ (df['Ypred'] == 1) & (df['Y'] == 1) ]    
    x2 = df[ (df['Ypred'] != df['Y'] ) ]    
    return x0, x1, x2
        



##Prob 1.2.c
# try a non linear l method
#define the dataset and get data.
def load_data(name, num):
    if name == 'classification':
        Xtrain, Xtest, Ytrain, Ytest = hw1.Make_classification(num).data_split()
        print(name)
    elif name == 'moons':
        Xtrain, Xtest, Ytrain, Ytest = hw1.Make_moons(num).data_split()
        print(name)
    else:
        Xtrain, Xtest, Ytrain, Ytest = hw1.Make_circles(num).data_split()
        print(name)
    return Xtrain, Xtest, Ytrain, Ytest

#define a function to show the loss history of neural network
def show_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

##define each dataset's network
class Network_model_classification(torch.nn.Module):
  def __init__(self):
    super(Network_model_classification, self).__init__()
    self.linear1 = torch.nn.Linear(2, 30, bias=True)
    self.linear2 = torch.nn.Linear(30, 20, bias=True)
    self.linear3 = torch.nn.Linear(20, 1, bias=True)

  def forward(self, x):
    x = nn.functional.relu(self.linear1(x))
    x = nn.functional.relu(self.linear2(x))
    x = self.linear3(x)
    x = torch.sigmoid(x)
    return x

class Network_model_moons(torch.nn.Module): 
  def __init__(self):
    super(Network_model_moons, self).__init__()
    self.linear1 = torch.nn.Linear(2, 30, bias=True)
    self.linear2 = torch.nn.Linear(30, 20, bias=True)
    self.linear3 = torch.nn.Linear(20, 1, bias=True)

  def forward(self, x):
    x = nn.functional.relu(self.linear1(x))
    x = nn.functional.relu(self.linear2(x))
    x = self.linear3(x)
    x = torch.sigmoid(x)
    return x

class Network_model_circles(torch.nn.Module):
  def __init__(self):
    super(Network_model_circles, self).__init__()
    self.linear1 = torch.nn.Linear(2, 30, bias=True)
    self.linear2 = torch.nn.Linear(30, 20, bias=True)
    self.linear3 = torch.nn.Linear(20, 1, bias=True)

  def forward(self, x):
    x = nn.functional.relu(self.linear1(x))
    x = nn.functional.relu(self.linear2(x))
    x = self.linear3(x)
    x = torch.sigmoid(x)
    return x



##define the training process
def train(network, name, optimizer, criterion,  n_epochs = 30, batch_size = 10, num = 1000):
    
    # construct corresponding data from 'name' dataset
    Xtrain, Xtest, Ytrain, Ytest = load_data(name, num)
    # change the X and Y data into tensor
    Xt = torch.tensor(Xtrain, dtype=torch.float)
    Yt = torch.tensor(Ytrain, dtype=torch.float).reshape(-1, 1)
    n_samples, n_features = Xtrain.shape

    history = []# define loss history
    training_accuracy = []
    validation_accuracy = []

    #begin training process
    for epoch in range(n_epochs):
        epoch_training_loss = 0.0# loss for each epoch
        permutation = torch.randperm(n_samples)

        for batch_start in range(0, n_samples, batch_size): # begin each batch

            idx = permutation[batch_start:batch_start + batch_size]

            Xbatch = Xt[idx, :]

            Ybatch = Yt[idx, :]

            forward_output = network( Xbatch ) 

            loss = criterion(forward_output, Ybatch ) # compute the loss based on givin criterion

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_training_loss += loss.item()

        # Ending of training process
        # Begining of accuracy process
        accuracy = 0.0
        num_batches = 0
        # calculate each accuracy for each batch
        for batch_start in range(0, n_samples, batch_size):
            idx = permutation[batch_start:batch_start + batch_size]
            Xbatch = Xt[idx, :]
            Ybatch = Yt[idx, :]
            num_batches += 1
            with torch.no_grad(): # the next line will not compute any gradient.
                Ypred = network(Xbatch).data.numpy()
            accuracy += accuracy_score( Ypred.round(), Ybatch.numpy())

        avg_train_accuracy = accuracy / num_batches# calculate the average accuracy for all batch
        training_accuracy.append(avg_train_accuracy)
        history.append(epoch_training_loss)

    return n_epochs, training_accuracy, history, Xtest, Ytest

##define the testing process
def test(network, Xtest, Ytest, batch_size = 10):
    n_samples, n_features = Xtest.shape
    Xt = torch.tensor(Xtest, dtype=torch.float)
    Yt = torch.tensor(Ytest, dtype=torch.float).reshape(-1, 1)
    num_batches =0
    accuracy =0
    permutation = torch.randperm(n_samples)
    testing_accuracy=0
    #compute each batch accuracy
    for batch_start in range(0, n_samples, batch_size):
        idx = permutation[batch_start:batch_start + batch_size]
        Xbatch = Xt[idx, :]
        Ybatch = Yt[idx, :]
        num_batches += 1
        with torch.no_grad(): # the next line will not compute any gradient.
            Ypred =  network(Xbatch).data.numpy()
        accuracy += accuracy_score( Ypred.round(), Ybatch.numpy())
    testing_accuracy = accuracy / num_batches# calculate the average accuracy for all batch
    print(f"Test accuracy: {testing_accuracy}")
    return Xtest, Ytest


## plotting the trainning accuracy figure
def plot_train_log(epochs, training_accuracy):
    pyplot.plot(epochs, training_accuracy, 'r')
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Accuracy")
    pyplot.show()

## plotting the seperating figures
def show_separation(model,X, y, save=False, name_to_save=""):
    sns.set(style="white")

    xx, yy = np.mgrid[-1.5:2.5:.01, -1.:1.5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    batch = torch.from_numpy(grid).type(torch.float32)
    with torch.no_grad():
        probs =  model(batch).reshape(xx.shape) 
        probs = probs.numpy().reshape(xx.shape)

    f, ax = plt.subplots(figsize=(16, 10))
    ax.set_title("Decision boundary", fontsize=14)
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(xlabel="$X_1$", ylabel="$X_2$")
    if save:
        plt.savefig(name_to_save)
    else:
        plt.show()
