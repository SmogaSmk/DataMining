import numpy as np 
import torch
import torch.nn as nn 
from joblib import Parallel, delayed
from abc import ABC 

# Parameterized module
class LinearMultiClassification(nn.Module): 
  def __init__(self, input_size, output_size, hidden_size1 = 10, hidden_size2 = 8): 
    super(LinearMultiClassification, self).__init__()
    
    self.input_size = input_size 
    self.output_size = output_size 
    self.hidden_size1 = hidden_size1 
    self.hidden_size2 = hidden_size2

    self.network = nn.Sequential(
      nn.Linear(input_size, hidden_size1), 
      nn.ReLU(), 
      nn.Linear(hidden_size1, hidden_size2), 
      nn.ReLU(), 
      nn.Linear(hidden_size2, output_size)
    )
    
  def forward(self, x): 
    return self.network(x)

  def __repr__(self): 
    return (f"LinearMultiClassification("
            f"input_size = {self.input_size}, "
            f"hidden_size1 = {self.hidden_size1}, "
            f"hidden_size2 = {self.hidden_size2}, "
            f"output_size = {self.output_size} )"
            )
  # =========== usage ============
  '''
  1. Training: 
  trainer = ClassificationTrainer(model, device = 'cuda')
  train_losses, val_accuracies = trainer.train(train_loader, val_loader, epochs = 50)
  
  2. Model Evaluation
  # evaluate on validation sets
  val_acc = ModelEvaluator.evaluate_classification_model(trainer.model, val_loader, trainer.device)

  # Final test
  trainer.model.load_state_dict(torch.load('best_model.pth'))
  test_acc = ModelEvaluator.evaluate_classification_model(trainer.model, test_loader, trainer.device)
  print(f'Final Test Accuracy: {test_acc:.2f}%')
  '''
