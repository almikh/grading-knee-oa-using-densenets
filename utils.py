import re
import os
import torch

def compute_accuracy(model, loader):
    model.eval() # Evaluation mode
    
    predictions = []
    raw_predictions = []
    ground_truth = []

    total_samples = 0
    correct_samples = 0
    for i_step, (x, y) in enumerate(loader):
        x_gpu = x
        y_gpu = y
        
        prediction = model(x_gpu) 

        indices = torch.argmax(prediction, 1)
        correct_samples += torch.sum(indices == y_gpu)
        total_samples += y.shape[0]
            
        # store result
        raw_predictions.extend(prediction.tolist())
        predictions.extend(indices.tolist())
        ground_truth.extend(y.tolist())

    return float(correct_samples) / total_samples, predictions, ground_truth, raw_predictions
