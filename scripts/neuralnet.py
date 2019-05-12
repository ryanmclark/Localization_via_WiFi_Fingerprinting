
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is for OMP error #15. You can comment it out in some environment.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

class NeuralNetworkRegressor(nn.Module):
    def __init__(self, D_in=520, H1=10, H2=5, D_out=2):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
        self.already_trained = False

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def fit(self, x_train, y_train):
        if self.already_trained:
            return
        else:
            self.already_trained = True
        epochs = 30
        batch_size = 100
        lr = 0.05
        x_train = np.array(x_train)
        y_train = np.array(y_train) 
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2, shuffle=True)
        # These x_val and y_val are different from the original validation data in UJIIndoorLoc
        x_train_tensor = torch.FloatTensor(x_train)
        y_train_tensor = torch.FloatTensor(y_train)
        x_val_tensor = torch.FloatTensor(x_val)
        y_val_tensor = torch.FloatTensor(y_val)
        dataset_train = TensorDataset(x_train_tensor, y_train_tensor)
        dataset_val = TensorDataset(x_val_tensor, y_val_tensor)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        #self.epoch_loss_history = []
        #self.val_epoch_loss_history = []
        for e in range(epochs):
            running_loss = 0.0
            val_running_loss = 0.0
            for x, label in dataloader_train:
                pred = self(x)
                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                with torch.no_grad():
                    for val_x, val_label in dataloader_val:
                        val_pred = self(val_x)
                        val_loss = criterion(val_pred, val_label)
                        val_running_loss += val_loss.item()
                epoch_loss = running_loss/len(dataloader_train)
                val_epoch_loss = val_running_loss/len(dataloader_val)
                #self.epoch_loss_history.append(epoch_loss)
                #self.val_epoch_loss_history.append(val_epoch_loss)
                print("epoch:{0: >3}  training loss:{1: >15}  validation loss:{2: >15}".format(e, int(epoch_loss), int(val_epoch_loss)))

    def predict(self, x_test):
        x_test_tensor = torch.FloatTensor(np.array(x_test))
        prediction = np.zeros((len(x_test), 2))
        for i,x in enumerate(x_test_tensor):
            prediction[i] = self(x).detach().numpy()
        return prediction

