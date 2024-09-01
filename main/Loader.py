import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
class Loader:
    def __init__(self,dataframeIn,featureNames,labelName):
        self.dataframeIn=dataframeIn
        self.featureNames=featureNames  
        self.labelName=labelName
        self.dataset=None
        self.features=None
        self.label=None

    def createDataloader(self,batch_size):
        self.features=torch.tensor(self.dataframeIn[self.featureNames].to_numpy()).float()
        self.label=torch.tensor(self.dataframeIn[self.labelName].to_numpy()).long()
        self.dataset=TensorDataset(self.features,self.label)
        dataloader=DataLoader(self.dataset,batch_size)
        return dataloader


# sample_np=np.random.rand(35,4)
# df=pd.DataFrame(sample_np)
# print(df)
# df.columns=['feature1','feature2','feature3','label']
# print(df)
# featureNames=['feature1','feature2','feature3']
# labelName='label'
# loader=Loader(df,featureNames,labelName)
# dataload=loader.createDataloader(32)
# print(dataload)

        
    