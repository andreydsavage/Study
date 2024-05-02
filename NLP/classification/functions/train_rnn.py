import numpy as np
import torch
from time import time

def train(
        epochs: int, 
        model, 
        train_loader,
        valid_loader, 
        optimizer, 
        rnn_conf,
        criterion,
        metric
        ) -> tuple: 
    """Training recurrent model for binary classification task

    Args:
        epochs (int): Number of epochs
        model (nn.Module): Model instance
        train_loader (Dataloader): train loader
        valid_loader (Dataloader): valid loader
        optimizer (nn.Module): optimizer
        rnn_conf (dataclass): dataclass with params
        criterion (nn.Module): criterion
        metric (_type_): metric from torchmetrics

    Returns:
        tuple: (train loss, valid loss, train metric, valid metric, training time)
    """
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_train_metric = []
    epoch_valid_metric = []
    time_start = time()
    for epoch in range(epochs):
        bacth_losses = []
        batch_metric = []
        model.train()
        model.to(rnn_conf.device)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rnn_conf.device), labels.to(rnn_conf.device)   

            output = model(inputs).squeeze()
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bacth_losses.append(loss.item())
            batch_metric.append(metric(output,labels).item())
        epoch_train_losses.append(np.mean(bacth_losses))
        epoch_train_metric.append(np.mean(batch_metric))
        bacth_losses = []
        batch_metric = []
        model.eval()
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(rnn_conf.device), labels.to(rnn_conf.device)
            with torch.no_grad():
                output = model(inputs).squeeze()
            loss = criterion(output, labels)
            bacth_losses.append(loss.item())
            batch_metric.append(metric(output.squeeze(),labels).item())
        epoch_valid_losses.append(np.mean(bacth_losses))
        epoch_valid_metric.append(np.mean(batch_metric))


        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_losses[-1]:.4f} val_loss : {epoch_valid_losses[-1]:.4f}')
        print(f'train_accuracy : {epoch_train_metric[-1]:.2f} val_accuracy : {epoch_valid_metric[-1]:.2f}')
            
        print(25*'==')
        training_time = time() - time_start
    return (epoch_train_losses, epoch_valid_losses, epoch_train_metric, epoch_valid_metric, training_time)
        