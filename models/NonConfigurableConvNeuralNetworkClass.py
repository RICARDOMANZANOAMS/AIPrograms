import torch
import torch.nn as nn
import numpy as np
class NonConfConvNeuralNetwork(nn.Module):
    def __init__(self, channels,output):
        self.output=output
        super(NonConfConvNeuralNetwork,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=10,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10,out_channels=3,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)           
        )

        self.stack=nn.Sequential(
            nn.Flatten())
    

    def forward (self,x):
        x=self.conv1(x)
        print("after conv")
        print(x.shape)
        x=self.stack(x)
        self.linear=nn.Linear(x.shape[1],self.output)
        x=self.linear(x)
        print("after flatten")
        print(x)
        print(x.shape)
   
        return x

network=NonConfConvNeuralNetwork(3,2)
print(network)
#dimensions means batch_size, channels, dim_x, dim_y
array=np.random.rand(1,3,10,12)
print(array)
tensor_array=torch.tensor(array).float()
print(tensor_array)

output=network(tensor_array)
print("output")
print(output)


