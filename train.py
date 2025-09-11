import torch 
import numpy as np
import torch.nn as nn 
import torch.optim as optim 

class ClassificationTrainer:
  def __init__(self, model, criterion=None, optimizer=None, device = 'cpu'): 
    self.model = model.to(device) 
    self.device = device
    self.criterion = criterion or nn.CrossEntropyLoss()
    self.optimizer = optimizer or optim.Adam(model.parameters(), lr = 0.001)
    self.best_val_acc = 0
    self.patience = 10
    self.patience_counter = 0
    
  def train(self, train_loader, val_loader=None, epochs = 100):
    train_losses = [] 
    val_accuracies = []
    
    for epoch in range(epochs): 
      # training 
      train_loss = self._train_epoch(train_loader)
      train_losses.append(train_loss)
      
      # validating
      val_acc = self.evaluate(val_loader)
      val_accuracies.append(val_acc)
      
      # pre_shutting_off
      if val_acc > self.best_val_acc: 
        self.best_val_acc = val_acc 
        self.patience_counter = 0
        
        torch.save(self.model.state_dict(), 'best_model.pth')
      else: 
        self.patience_counter += 1

      if self.patience_counter >= self.patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
      
    return train_losses, val_accuracies
  
  def _train_epoch(self, train_loader):
    self.model.train() 
    total_loss = 0
    
    for data, target in train_loader: 
      data, target = data.to(self.device), target.to(self.device)
      
      self.optimizer.zero_grad() 
      output = self.model(data)
      loss = self.criterion(output, target) 
      loss.backward() 
      self.optimizer.step()
      
      total_loss += loss.item()
      
    return total_loss / len(train_loader)
