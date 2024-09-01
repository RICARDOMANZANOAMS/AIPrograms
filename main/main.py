        
from NonConfigurableNeuralNetworkClass import NeuralNetwork
from Trainer import Trainer

model=NeuralNetwork(4,20,2)
print(model) 
from Split import Split
from Loader import Loader
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
x=np.random.rand(20,4)
dfx=pd.DataFrame(x)
y=np.random.choice([0,1],20)
dfy=pd.DataFrame(y)
dataset=pd.concat([dfx,dfy],axis=1)

dataset.columns=['f1','f2','f3','f4','label']
split=Split(dataset)
partitions=split.trainTestSplit(0.2)
featureNames=['f1','f2','f3','f4']
labelName='label'
print("partitions")
print(partitions)
print("end")
trainer=Trainer()
for partition in partitions:
    # print("partition")
    # print(partition)
    for i in range(2):
        if i==0:
            train=partition[i]
            #print(train)
            loader=Loader(train,featureNames,labelName)
            dataloaderTrain=loader.createDataloader(32)
            print("train")
            print(dataloaderTrain)
            loss=nn.CrossEntropyLoss()
            optimizer=optim.Adam(model.parameters(),lr=0.00001)
            trainer.trainAlg(dataloaderTrain,model,loss,optimizer)
            trainer.saveModel(trainer.model,'model2.pth')

        if i==1:
            test=partition[i]
            #print(test)
            loader=Loader(test,featureNames,labelName)
            dataloaderTest=loader.createDataloader(32)
            model=trainer.loadModel('model2.pth')
            print("test")
            print(dataloaderTest)
            trainer.testAlg(dataloaderTest,model)

 