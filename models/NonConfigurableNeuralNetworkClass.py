import torch 
import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self, inputLayerDim,hiddenLayerDim,outputLayerDim):
        super(NeuralNetwork,self).__init__()
        self.hidden=nn.Linear(inputlayerDim,hiddenLayerDim)
        self.output=nn.Linear(hiddenLayerDim,outputLayerDim)
    def forward(self,x):
        x=torch.relu(self.hidden(x))
        x=self.output(x)
        return x

        
        