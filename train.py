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
  
  def evaluate(self, data_loader):
    self.model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(): 
      for data, target in data_loader: 
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0) 
        correct += (predicted == target).sum().item()
        
    return 100 * correct / total 
  
  def final_test(self, test_loader): 
    self.model.load_state_dict(torch.load('best_model.pth'))
    test_acc = self.evaluate(test_loader)
    print(f'Final Test Accuracy: {test_acc:.2f}%')
    return  test_acc

class KNNTrainer: 
  def __init__(self):
    pass 
  
  def train(self, model, train_loader, test_loader = None, **kwargs): 
    '''
    train the model: 
    :Args:
      model: the model to train (usually KNN model) 
      train_loader: a loader that load data for training
      test_loader(optional): a loader prepare for data evaluating the model's performance
      **kwargs: 
        - 
    '''
    
    X_train_list = []
    y_train_list = []
    
    for batch_x, batch_y in train_loader: 
      X_train_list.append(batch_x)
      y_train_list.append(batch_y)
      
    if torch.is_tensor(X_train_list[0]): 
      X_train = torch.get(X_train_list, dim = 0)
      y_train = torch.get(y_train_list, dim = 0)   
    else: 
      X_train = np.vstack(X_train_list)
      y_train = np.hstack(y_train_list)
      
    model.fit(X_train, y_train)
    
    test_metric = {}
    if test_loader is not None: 
      test_metrics = self.test(model, test_loader)
      print(f"examine metrics: {test_metrics}")
      
    return model, test_metrics
  
  def test(self, model, test_loader): 
    '''evaluate the model's performance'''
    all_predictions = [] 
    all_targets = [] 
    
    for batch_x, batch_y in test_loader: 
      predictions = model.predict(batch_x)
      
      if torch.is_tensor(predictions): 
        predictions = predictions.numpy()
      if torch.is_tensor(batch_y):
        batch_y = batch_y.numpy()
        
      all_predictions.extend(predictions)
      all_targets.extend(batch_y)
      
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # calculate metrics 
    if model.task_type == 'classification': 
      accuracy = np.mean(all_predictions == all_targets)
      return {'accuracy': accuracy}
    else: 
      mse = np.mean((all_predictions - all_targets) ** 2)
      return {'mse': mse}
    
    
  