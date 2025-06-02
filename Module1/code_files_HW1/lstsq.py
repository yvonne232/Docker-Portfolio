import numpy as np
import torch

def fit(X, Y, n_iter=10, batch_size=10, eta=0.1, w=None):
  n_samples, n_features = X.shape

  if w is None:
    w = torch.zeros(n_features, requires_grad=True, dtype=torch.float)
  else:
    assert len(w.shape) == n_features

  # convert numpy arrays to torch arrays
  Xt = torch.tensor(X, dtype=torch.float)
  Yt = torch.tensor(Y, dtype=torch.float)

  history = []
  
  optimizer = torch.optim.SGD([w], lr=eta)

  # epochs
  for i in range(n_iter):
    
    permutation = torch.randperm(n_samples)

    epoch_loss = 0
    
    for batch_start in range(0, n_samples, batch_size):

      idx = permutation[batch_start:batch_start + batch_size]
      Xbatch = Xt[idx, :]
      Ybatch = Yt[idx]
  
      # mv = matrix-vector multiplication in Torch
      Ypred = Xbatch.mv(w)
      
      loss_batch = torch.sum((Ypred - Ybatch)**2) / n_samples
      
      # .item() converts a 1-element tensor to a Python float
      epoch_loss += loss_batch.item()
      
      # reset gradients from previous batch
      optimizer.zero_grad()

      # compute the gradients for this batch
      loss_batch.backward()

      # update the parameters
      # for SGD, this is equivalent to w -= learning_rate * gradient
      optimizer.step()
                
    history.append(epoch_loss)

  return w, history
