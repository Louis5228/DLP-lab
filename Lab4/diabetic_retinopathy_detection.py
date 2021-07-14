# -*- coding: utf-8 -*-
"""Diabetic Retinopathy Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z4ReqMFATYO-Wnn7HbuE8iWsAXr1Qt88
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd DLP-lab/Lab4/

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from dataloader import RetinopathyLoader

class ResNet18(nn.Module):
  def __init__(self, num_class, pretrained=False):
    super(ResNet18, self).__init__()
    self.model = models.resnet18(pretrained=pretrained)
    if pretrained:
      for param in self.model.parameters():
        param.requires_grad = False
    num_neurons = self.model.fc.in_features
    self.model.fc = nn.Linear(num_neurons, num_class)
      
  def forward(self, x):
    out = self.model(x)
    return out

class ResNet50(nn.Module):
  def __init__(self, num_class, pretrained=False):
    super(ResNet50, self).__init__()
    self.model = models.resnet50(pretrained=pretrained)
    if pretrained:
      for param in self.model.parameters():
        param.requires_grad = False
    num_neurons = self.model.fc.in_features
    self.model.fc = nn.Linear(num_neurons, num_class)
      
  def forward(self, x):
    out = self.model(x)
    return out

def evaluate(model, loader_test, device, num_class):
  confusion_matrix = np.zeros((num_class, num_class))

  with torch.set_grad_enabled(False):
    correct = 0
    for _, data in enumerate(loader_test, 0):
      inputs = data[0].to(device)
      labels = data[1].to(device, dtype=torch.long)
      predict = model(inputs)
      predict_class = predict.max(dim=1)[1]
      correct += predict_class.eq(labels).sum().item()
      for i in range(len(labels)):
        confusion_matrix[int(labels[i])][int(predict_class[i])] += 1
    correct = (correct / len(loader_test.dataset)) * 100.0
  
  confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1).reshape(num_class, 1)

  return confusion_matrix, correct

def train(model, loader_train, loader_test, num_class, epochs, optimizer, criterion, device, name):
  print("Start training {} weights".format(name))
  df = pd.DataFrame()
  df['epoch'] = range(1, epochs + 1)
  best_model_wts = None
  best_evaluated_acc = 0
  model.to(device)
  train_acc = []
  test_acc = []

  for epoch in range(1, epochs+1):
    ## train
    with torch.set_grad_enabled(True):
      model.train()
      running_loss = 0
      correct = 0
      for i, data in enumerate(loader_train, 0):
        # get the inputs
        inputs = data[0].to(device)
        labels = data[1].to(device, dtype=torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predict = model(inputs)
        loss = criterion(predict, labels)
        running_loss += loss.item()
        correct += predict.max(dim=1)[1].eq(labels).sum().item()
        loss.backward()
        optimizer.step()

      running_loss /= len(loader_train.dataset)
      train_correct = (correct / len(loader_train.dataset)) * 100.0
      train_acc.append(train_correct)

    ## test
    model.eval()
    _, test_correct = evaluate(model, loader_test, device, num_class)
    test_acc.append(test_correct)

    print("[Train] epcoh{:>4d}  loss:{:.5f}  acc:{:.2f}%    [Test] acc:{:.2f}%".format(epoch, running_loss, train_correct, test_correct))

    # if test_correct > best_evaluated_acc:
    #   best_evaluated_acc = test_correct
    #   best_model_wts = copy.deepcopy(model.state_dict())
    
    # torch.save(best_model_wts,os.path.join('models', name + '.pt'))
    # model.load_state_dict(best_model_wts)

  df['train_acc'] = train_acc
  df['test_acc'] = test_acc
  print("Finished training {} weights".format(name))

  return df

def plot_confusion_matrix(matrix):
  fig, ax = plt.subplots(figsize=(6,6))
  ax.matshow(matrix, cmap=plt.cm.Blues)
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      ax.text(i, j, '{:.2f}'.format(matrix[j, i]), va='center', ha='center')
  ax.set_xlabel('Predicted label')
  ax.set_ylabel('True label')
  return fig

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""## ResNet18"""

num_class = 5
# batch_size = 8
lr = 0.005
epochs = 24
epochs_extraction = 8
epochs_finetune = epochs - epochs_extraction
momentum = 0.9
weight_decay = 5e-4
criterion = nn.CrossEntropyLoss()

dataset_train = RetinopathyLoader(root = './data', mode = 'train')
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

dataset_test = RetinopathyLoader(root = './data', mode = 'test')
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

'''
ResNet18 w/o pretrained weights
'''
model_wo = ResNet18(num_class=num_class, pretrained=False)
optimizer = optim.SGD(model_wo.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
df_wo_pretrained = train(model_wo, loader_train, loader_test, num_class, epochs, optimizer, criterion, device, 'ResNet18 w/o pretrained')

# get a confusion matrix
confusion_matrix, _ = evaluate(model_wo, loader_test, device, num_class)
figure = plot_confusion_matrix(confusion_matrix)
# figure.savefig('images/ResNet18(w/o pretrained weights).png')

'''
ResNet18 with pretrained weights
'''
model_with = ResNet18(num_class=num_class, pretrained=True)

# feature extraction
params_to_update = []
for _, param in model_with.named_parameters():
  if param.requires_grad:
    params_to_update.append(param)

optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
df_extraction = train(model_with, loader_train, loader_test, num_class, epochs_extraction, optimizer, criterion, device, 'ResNet18 with pretrained')

# finetuning
for param in model_with.parameters():
  param.requires_grad = True

optimizer = optim.SGD(model_with.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
df_finetune = train(model_with, loader_train, loader_test, num_class, epochs_finetune, optimizer, criterion, device, 'ResNet18 with pretrained')

df_with_pretrained = pd.concat([df_extraction, df_finetune], axis=0, ignore_index=True)

# get a confusion matrix
confusion_matrix, _ = evaluate(model_with, loader_test, device, num_class)
figure = plot_confusion_matrix(confusion_matrix)
# figure.savefig('images/ResNet18(with pretrained weights).png')

plt.figure(figsize=(10, 6))

# for name in df_wo_pretrained.columns[1:]:
#   plt.plot(range(1, 1 + len(df_wo_pretrained)), name, data=df_wo_pretrained, label=name[4:]+'(w/o pretraining)')
for name in df_with_pretrained.columns[1:]:
  plt.plot(range(1, 1 + len(df_with_pretrained)), name, data=df_with_pretrained, label=name[4:]+'(with pretraining)')

plt.title("Result Comparison(ResNet18)", fontsize = 15)
plt.xlabel("Epochs", fontsize = 12)
plt.ylabel("Accuracy(%)", fontsize = 12)
plt.legend()
plt.show()

"""## ResNet50"""

batch_size = 6

dataset_train = RetinopathyLoader(root = './data', mode = 'train')
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

dataset_test = RetinopathyLoader(root = './data', mode = 'test')
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

'''
ResNet50 w/o pretrained weights
'''
model_wo = ResNet50(num_class=num_class, pretrained=False)
optimizer = optim.SGD(model_wo.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
df_wo_pretrained = train(model_wo, loader_train, loader_test, num_class, epochs, optimizer, criterion, device, 'ResNet50 w/o pretrained')

# get a confusion matrix
confusion_matrix, _ = evaluate(model_wo, loader_test, device, num_class)
figure = plot_confusion_matrix(confusion_matrix)
# figure.savefig('/home/louis/DLP-lab/Lab4/images/ResNet50(w/o pretrained weights).png')

'''
ResNet50 with pretrained weights
'''
model_with = ResNet50(num_class=num_class, pretrained=True)

# feature extraction
params_to_update = []
for _, param in model_with.named_parameters():
  if param.requires_grad:
    params_to_update.append(param)

optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
df_extraction = train(model_with, loader_train, loader_test, num_class, epochs_extraction, optimizer, criterion, device, 'ResNet50 with pretrained')

# finetuning
for param in model_with.parameters():
  param.requires_grad = True

optimizer = optim.SGD(model_with.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
df_finetune = train(model_with, loader_train, loader_test, num_class, epochs_finetune, optimizer, criterion, device, 'ResNet50 with pretrained')

df_with_pretrained = pd.concat([df_extraction, df_finetune], axis=0, ignore_index=True)

# get a confusion matrix
confusion_matrix, _ = evaluate(model_with, loader_test, device, num_class)
figure = plot_confusion_matrix(confusion_matrix)
# figure.savefig('/home/louis/DLP-lab/Lab4/images/ResNet50(with pretrained weights).png')

plt.figure(figsize=(10, 6))

for name in df_wo_pretrained.columns[1:]:
  plt.plot(range(1, 1 + len(df_wo_pretrained)), name, data=df_wo_pretrained, label=name[4:]+'(w/o pretraining)')
for name in df_with_pretrained.columns[1:]:
  plt.plot(range(1, 1 + len(df_with_pretrained)), name, data=df_with_pretrained, label=name[4:]+'(with pretraining)')

plt.title("Result Comparison(ResNet50)", fontsize = 15)
plt.xlabel("Epochs", fontsize = 12)
plt.ylabel("Accuracy(%)", fontsize = 12)
plt.legend()
plt.show()