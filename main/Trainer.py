
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
class Trainer():
    def __init__(self):
        self.model=None
    
    def trainAlg(self,loaderTrain,model,loss,optimizer):
        for features, labels in loaderTrain:
            optimizer.zero_grad()
            output=model(features)
            loss=loss(output,labels)
            loss.backward()
            optimizer.step()
        self.model=model
      
            

    def testAlg(self,loaderTest,model):
        predicted=[]
        true=[]
        for features, labels in loaderTest:
            with torch.no_grad():
                output=model(features)
            probs=nn.functional.softmax(output,dim=1)
            _,labelsPred=torch.max(probs,1)
            labelsPred=labelsPred.data.cpu().numpy().tolist()
            labelsTrue=labels.data.cpu().numpy().tolist()
            
            predicted.append(labelsPred)
            true.append(labelsTrue)
       
        predicted = [item for sublist in predicted for item in sublist]
        true = [item for sublist in true for item in sublist]
      
        cm=confusion_matrix(predicted,true)
        print(cm)

    @staticmethod
    def saveModel(model,path):
        torch.save(model,path)

    @staticmethod
    def loadModel(path):
        model=torch.load(path)
        return model



    
