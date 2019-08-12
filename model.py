import torch 
from torchvision import models, transforms
import torch.nn as nn 
from torch.utils.data import DataLoader
from Dataset import ChestXRay
import torch.optim as optim
#import matplotlib.pyplot as plt
import time 
import copy
#from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, roc_auc_score, classification_report
import csv 
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
n_classes = 14

train_file = 'demo_list.txt'
val_file = 'val_demo_list.txt'
test_file = 'test_demo_list.txt'

class DenseNet121(nn.Module):
    def __init__(self, n_classes):
        super(DenseNet121, self).__init__()
        
        self.densenet121 = models.densenet121(pretrained=True)
        n_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
                nn.Linear(n_features, n_classes),
                nn.Sigmoid()
                )
        
    def forward(self, x):
        x = self.densenet121(x)
        return x
    

    
    #The Transforms dict
dataset_transforms = {'train': transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.RandomHorizontalFlip(),                                                       
                                                   transforms.ToTensor()]),
                        'val': transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()])}
    
    #Transformed Datasets
datasets = {}
datasets['train'] = ChestXRay(train_file, dataset_transforms['train']) 
datasets['val'] = ChestXRay(val_file, dataset_transforms['val'])
    
    #Data Loaders
data_loaders = {}
data_loaders['train'] = DataLoader(datasets['train'], batch_size=2, shuffle=True)
data_loaders['val'] = DataLoader(datasets['val'], batch_size=1, shuffle=True)
    
dataset_lengths = {x : len(data_loaders[x]) for x in ['train', 'val']}
    
model =  DenseNet121(n_classes)
    
criterion = nn.BCELoss()
    
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

scheduler = ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=1)
#maxEpoch = 4

train_loss = [] 
loss_val = []
def epoch_train(model, dataset_lengths, data_loaders, criterion, optimizer, epoch, maxEpoch):
    
    print("*"*50)
    print("Training Phase")
    print("*"*50)
    for batch_ids, (inputs, labels) in enumerate(data_loaders['train'], 1):
            
        outputs = model(inputs)
        
        optimizer.zero_grad()
        
        loss = criterion(outputs, labels)
        
        print("Epoch: {} Batch: {} Training Loss: {}".format(epoch, batch_ids, loss.item()))
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss)
        
    inputs = None
    labels = None    
        
val_loss = 0.0  
      
def epoch_validate(model, dataset_lengths, data_loaders, criterion, optimizer, epoch, maxEpoch):
    
    loss_mean = 0.0
    
    print("*"*50)
    print("Validation Phase")
    print("*"*50)
    
    for batch_ids, (inputs, labels) in enumerate(data_loaders['val'], 1):
        
        outputs = model(inputs)
        
        preds = outputs > 0.5
        
        loss = criterion(outputs, labels)
        print("Epoch: {} Batch: {} Running Loss: {}".format(epoch, batch_ids, loss.item()))

        loss_mean += loss.item()
        
        loss_val.append(loss)
        
    val_loss = loss_mean/dataset_lengths['val']
    
    #with open('validation_losses.txt', 'a+') as loss:
        #loss.write(str(val_loss))
        
        
    print("Epoch: {} Validation Loss: {}".format(epoch, val_loss))
    
    return val_loss

def compute_auc_roc(dataGT, dataPRED, classCount):
         
    outAUROC = []
        
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    
    #print(datanpGT[0])
    #print(datanpPRED[0])
    
    #print(classification_report(datanpGT[0], datanpPRED[0]))
    
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
        return outAUROC


def test(model, criterion, optimizer):
    
    outGT = torch.FloatTensor()
    outPred = torch.FloatTensor()
    
    
    print("Loading Checkpoint....")
    
    model_ckpt = torch.load('model-01072019-223548.pt')
    model.load_state_dict(model_ckpt['model_state_dict'])
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    test_transforms = transforms.Compose([transforms.Resize(224), 
                                          transforms.TenCrop(224),
                                          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                          transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))])
    
    test_dataset = ChestXRay(test_file, test_transforms)
    
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    model.eval()
    
    for batch_id, (inp, labels) in enumerate(test_loader):
        
        outGT = torch.cat((outGT, labels), 0)
        
        bs, crops, c, h, w = inp.size()
        
        var_inp = inp.view(-1, c, h, w)
        
        out = model(var_inp)
        print("Batch: {}".format(batch_id))
        outMean = out.view(bs, crops, -1).mean(1)
        
        outPred = torch.cat((outPred, outMean.data), 0)

    aurocIndividual = compute_auc_roc(outGT, outPred, n_classes)
    
    aurocMean = np.array(aurocIndividual).mean()

    print ('AUROC mean: ', aurocMean)
        
    for i in range (0, len(aurocIndividual)):
        print (CLASS_NAMES[i], ' ', aurocIndividual[i])                    
                
        
        
"""    
with torch.no_grad():    
    test(model, criterion, optimizer)        
"""

lossMin = 100000

train_start = time.time()
for epoch in range(1, 11):
    
    print("Epoch: {}/{}".format(epoch,10))
     
    start = time.time()
    epoch_train(model, dataset_lengths, data_loaders, criterion, optimizer, epoch, 10)
    
    with torch.no_grad():
        loss = epoch_validate(model, dataset_lengths, data_loaders, criterion, optimizer, epoch, 10)
        
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampEND = timestampDate + '-' + timestampTime
        
        if loss < lossMin:
            lossMin = loss
            torch.save({'epoch': epoch, 'best_loss': lossMin, 'model': model, 'model_state_dict': model.state_dict()}, 'model-' + timestampEND + '.pt')
            
        scheduler.step(loss)
        
    print("Epoch completed in: {} seconds".format(time.time()-start))
    #print(torch.load('model-3.pth.tar'))
    
print("Training completed in: {} mins".format((time.time()-train_start)/60))

          
        
