import torch 
import numpy as np
import torch.nn as nn 
import torch.optim as optim 

class ClassificationTrainer:
  def __init__(self, model, criterion=None, optimizer=None, device = 'cpu'): 
    self.model = model.to(device) 
    self.device = device
    
    # 自动检测并应用模型的类别权重
    if criterion is None:
      class_weights = None
      if hasattr(model, 'get_class_weights'):
        class_weights = model.get_class_weights(device=device)
        if class_weights is not None:
          print(f"检测到类别权重: {class_weights}")
          print("应用加权交叉熵损失函数以处理不平衡数据")
        else:
          print("未检测到类别不平衡，使用标准交叉熵损失函数")
      
      self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
      self.criterion = criterion
      
    self.optimizer = optimizer or optim.Adam(model.parameters(), lr = 0.001)
    self.best_val_acc = 0
    self.patience = 10
    self.patience_counter = 0
    
  def evaluate(self, val_loader): 
    if val_loader is None: 
      return 0.0 
    
    self.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for data, target in val_loader: 
        data, target = data.to(self.device), target.to(self.device)
        
        if target.dtype != torch.long: 
          target = target.long()
          
        output = self.model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
    accuracy = 100 * correct / total 
    return accuracy
    
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
        
        best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
      else: 
        self.patience_counter += 1

      if self.patience_counter >= self.patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
      
    if best_model_state is not None: 
      self.model.load_state_dict(best_model_state)
      print("Load best model weights (validation accuracy: {self.best_val_acc:.4f})")
    return train_losses, val_accuracies
  
  def _train_epoch(self, train_loader):
    self.model.train() 
    total_loss = 0
    model_initialized = False
    
    for batch_idx, (data, target) in enumerate(train_loader): 
      try: 
        data, target = data.to(self.device), target.to(self.device)
    
        if target.dtype != torch.long:
          target = target.long()
        if batch_idx == 0: 
          print(f"Data shape: {data.shape}, Target shape: {target.shape}")
          print(f"Target dtype: {target.dtype}")
          print(f"Target range: [{target.min().item()}, {target.max().item()}]")
        
        self.optimizer.zero_grad()
        
        # 支持自动初始化的模型
        if hasattr(self.model, '_is_initialized') and not self.model._is_initialized:
          num_classes = int(target.max().item()) + 1
          output = self.model(data, num_classes=num_classes)
          if not model_initialized:
            # 重新创建优化器，因为模型参数可能已经改变
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            model_initialized = True
            # print(f"Model auto-initialized with input_size={self.model.input_size}, output_size={self.model.output_size}")
        else:
          output = self.model(data)
        
        if batch_idx == 0: 
          # print(f"Model output shape: {output.shape}")
          num_classes = output.shape[1] if len(output.shape) > 1 else output.shape[0]
          if target.max() >= num_classes:
            raise ValueError(f"Target label {target.max()} >= num_classes {num_classes}")
          if target.min() < 0: 
            raise ValueError(f"Target label {target.min()} < 0")
          
        if torch.isnan(output).any() : 
          print(f"Nan detected in output at batch {batch_idx}")
          continue
        if torch.isinf(output).any(): 
          print(f"Inf found in output at batch {batch_idx}")
          continue
        loss = self.criterion(output, target)
        
        # 如果模型有L2正则化方法（如Ridge回归），添加正则化项
        if hasattr(self.model, 'get_l2_penalty'):
          l2_penalty = self.model.get_l2_penalty()
          loss = loss + l2_penalty

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
        
        self.optimizer.step() 
        total_loss += loss.item() 
        
      except RuntimeError as e: 
        print(f"Error at batch {batch_idx}: {e}")
        print(f"Data shape: {data.shape if 'data' in locals() else 'N/A'}")
        print(f"Target shape: {target.shape if 'target' in locals() else 'N/A'}")
        print(f"Target values: {target if 'target' in locals() else 'N/A'}")
        raise e 
    
    '''for data, target in train_loader: 
      data, target = data.to(self.device), target.to(self.device)
      
      self.optimizer.zero_grad() 
      output = self.model(data)
      loss = self.criterion(output, target) 
      loss.backward() 
      self.optimizer.step()
      
      total_loss += loss.item()'''
      
    return total_loss / len(train_loader)
