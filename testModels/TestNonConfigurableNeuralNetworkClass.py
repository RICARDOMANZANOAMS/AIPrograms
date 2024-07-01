import numpy as np
import torch

import sys
import os
from torch.utils.data import TensorDataset, DataLoader
#Add models folder to the test run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from NonConfigurableNeuralNetworkClass import NeuralNetwork

features=4  #number of feautures
samples=10  #number of samples
array=np.random.rand(samples,features)  #generate numpy array
print(array)
features_torch=torch.tensor(array).float()  #transform to tensor
print(features_torch)

model=NeuralNetwork(features,4,2)  #create model
print(model)
output=model(features_torch)  
print(output)