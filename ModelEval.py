from sklearn.metrics import confusion_matrix
import seaborn as sn
import os
import matplotlib.pyplot as plt
import os
import random
import shutil
import pandas as pd
import numpy as np
import torch

# Generates a confusion matrix for a trained model
#
# model: trained pytorch model
# data_loader: pytorch dataloader used for confusion matrix generation
# train_dir_path: path to the training data directory that was used to train the model
# fig_size: normal matplotlib plot dimensions
def generate_confusion_matrix(model, data_loader, train_dir_path, fig_size=(96,56)):
    model.eval()
    y_pred = []
    y_true = []
    n_samples = len(data_loader)
    s = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for inputs, labels in data_loader:
            s+=1
            inputs = inputs.to(device)
            output = model(inputs)

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

            y_pred.extend(output) 

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) 
            if s % 1000 == 0:
                print(f"{s}/{n_samples}")
    
    classes = os.listdir(train_dir_path)
    if ".ipynb_checkpoints" in classes:
        classes.remove(".ipynb_checkpoints")

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = fig_size)
    sn.heatmap(df_cm, annot=True)
    plt.show()

        
        
# Test a given model on a dataset
#
# model: pytorch model that you want to test
# data_loader: pytorch dataloader with data that you want to test the model on
# top3top5: if True, function will also return the top3 and top5 accuracy
def test_model(model, data_loader, top3top5=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    top3correct = 0
    top5correct = 0
    top10correct= 0
    total = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward(inputs)
        _, predicted = outputs.topk(k=10, dim=1)
        total += targets.size(0)
        correct += predicted[:, 0].eq(targets).sum().item()
        if top3top5:
            top3correct += np.any(predicted.cpu().numpy()[:, :3] == np.expand_dims(targets.cpu().numpy(), axis=-1), axis=1).sum()
            top5correct += np.any(predicted.cpu().numpy()[:, :5] == np.expand_dims(targets.cpu().numpy(), axis=-1), axis=1).sum()
            top10correct +=np.any(predicted.cpu().numpy() == np.expand_dims(targets.cpu().numpy(), axis=-1), axis=1).sum()
    accuracy = (correct/total)*100
    print(f"Model accuracy: {correct}/{total} ===> {accuracy}%")
    if top3top5:
        top3accuracy = (top3correct/total)*100
        top5accuracy = (top5correct/total)*100
        top10accuracy= (top10correct/total)*100
        print(f"Top 3 accuracy: {top3correct}/{total} ===> {top3accuracy}%")
        print(f"Top 5 accuracy: {top5correct}/{total} ===> {top5accuracy}%")
        print(f"Top 10 accuracy: {top10correct}/{total} ==> {top10accuracy}%")
    
