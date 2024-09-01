from sklearn.model_selection import train_test_split,KFold
import numpy as np
import pandas as pd
class Split:
    def __init__(self,dataframe):
        self.dataframe=dataframe

    def trainTestSplit(self,test_size):
        partitions=train_test_split(self.dataframe,test_size=test_size,random_state=32)
        return [partitions]
    
    def kfold(self,numberOfK):
        partitions=[]
        kf=KFold(n_splits=numberOfK,shuffle=True,random_state=32)
        
        for train_idx, test_idx in kf.split(self.dataframe):
            partitionsIn=[]
            print(train_idx)
            print(test_idx)
            train_fold=self.dataframe.iloc[train_idx]
            test_fold=self.dataframe.iloc[test_idx]
            # print("train")
            # print(train_fold)
            # print("test")
            # print(test_fold)
            partitionsIn.append(train_fold)
            partitionsIn.append(test_fold)
            partitions.append(partitionsIn)
        return partitions
       
# x=np.random.rand(30,4)
# x=pd.DataFrame(x)
# split=Split(x)
# #partitions=split.trainTestSplit(0.3)

# partitions=split.kfold(3)
# print(partitions)
# print(len(partitions))
# for i in partitions:
#     print(len(i))

        