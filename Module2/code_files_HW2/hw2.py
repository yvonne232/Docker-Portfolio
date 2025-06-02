import sklearn.datasets
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import random
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('default')
from matplotlib import pyplot
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
from matplotlib.patches import Ellipse, Circle
###dataest making
def make_dataset(version=None, test=False):
    if test:
        random_state = None
    else:
        random_states = [27,33,38]
        if version is None:
            version = random.choice(range(len(random_states)))
            print(f"Dataset number: {version}")
        random_state = random_states[version]
    return sklearn.datasets.make_circles(factor=0.7, noise=0.1, random_state=random_state)


##define each dataset's network
class Neural_Network_98(torch.nn.Module):
  def __init__(self):
    super(Neural_Network_98, self).__init__()
    self.linear1 = torch.nn.Linear(2, 50, bias=True)
    self.linear2 = torch.nn.Linear(50, 40, bias=True)
    self.linear3 = torch.nn.Linear(40, 30, bias=True)
    self.linear4 = torch.nn.Linear(30, 20, bias=True)
    self.linear5 = torch.nn.Linear(20, 10, bias=True)
    self.linear6 = torch.nn.Linear(10, 1, bias=True)

  def forward(self, x):
    x = nn.functional.relu(self.linear1(x))
    x = nn.functional.relu(self.linear2(x))
    x = nn.functional.relu(self.linear3(x))
    x = nn.functional.relu(self.linear4(x))
    x = nn.functional.relu(self.linear5(x))
    x = self.linear6(x)
    x = torch.sigmoid(x)
    return x

class Neural_Network_Circle(torch.nn.Module):
  def __init__(self):
    super(Neural_Network_manully, self).__init__()
    self.linear1 = torch.nn.Linear(2, 30, bias=True)
    self.linear2 = torch.nn.Linear(30, 20, bias=True)
    self.linear3 = torch.nn.Linear(20, 1, bias=True)

  def forward(self, x):
    x = nn.functional.relu(self.linear1(x))
    x = nn.functional.relu(self.linear2(x))
    x = self.linear3(x)
    x = torch.sigmoid(x)
    return x

class Neural_Network_manully(torch.nn.Module):
  def __init__(self):
    super(Neural_Network_manully, self).__init__()
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
def train(network, Xtrain, Ytrain, optimizer, criterion,  n_epochs = 30, batch_size = 10, num = 1000):
    
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

    return n_epochs, training_accuracy, history

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
#define a function to show the loss history of neural network
def show_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

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

    ax.scatter(X[:,0], X[:, 1], c=y[100:], s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    
    cir1 = Circle(xy = (0.0, 0.0), radius=0.85, alpha=0.5)
    ax.add_patch(cir1)

    ax.set(xlabel="$X_1$", ylabel="$X_2$")
    if save:
        plt.savefig(name_to_save)
    else:
        plt.show()



#collect the Xtest data which Ypred == 1
def collect(x, yd, y):
    df = pd.DataFrame( data = x, columns = ['x0', 'x1']  )
    df['Ypred'] = yd
    df['Y'] = y
    x0 = df[ (df['Ypred'] == 0 )& (df['Y'] == 0 )]
    x1 = df[ (df['Ypred'] == 1) & (df['Y'] == 1) ]    
    x2 = df[ (df['Ypred'] != df['Y'] ) ]    
    return x0, x1, x2


##define the training process
def train_manully(network, optimizer, criterion,  n_epochs = 30, batch_size = 10, num = 1000):

    history = []# define loss history
    training_accuracy = []
    validation_accuracy = []

    #begin training process
    for epoch in range(n_epochs):
        #generating dataset
        Xtrain, Ytrain = make_dataset()
        # change the X and Y data into tensor
        Xt = torch.tensor(Xtrain, dtype=torch.float)
        Yt = torch.tensor(Ytrain, dtype=torch.float).reshape(-1, 1)
        n_samples, n_features = Xtrain.shape


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

    return n_epochs, training_accuracy, history
