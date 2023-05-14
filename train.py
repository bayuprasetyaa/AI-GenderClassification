import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

def loop_fn(mode, dataloader, model,
            criterion, optimizer, device, focus_on):
    '''
    '''
    CM = 0
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.eval()

    cost = correct = 0

    for feature, target in tqdm(dataloader, desc = mode.title()):
    # for feature, target in tqdm(dataloader):
        feature, target = feature.to(device), target.to(device)
        ## Feed Forward
        output = model(feature) ## Train model
        if mode == 'train':
            output = output.logits
        loss = criterion(output, target) ## get loss val
        print(output)

        if mode == 'train':
        ## Backpropagation
            loss.backward() ## get gradient to optimize weight val
            optimizer.step() ## update weight
            optimizer.zero_grad() ## reset weight

        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()

        preds = torch.argmax(output.data, 1)
        CM += confusion_matrix(target.cpu(), preds.cpu(), labels = [0, 1])
        print(CM)
  
    print(CM)
    tn = CM[0,0]
    tp = CM[1,1]
    fp = CM[0,1]
    fn = CM[1,0]

    # acc_new = np.sum(np.diag(CM)/np.sum(CM))
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    f1_ = ((2*recall*precision)/(recall+precision))

    cost = cost / len(dataloader)
    acc = correct / len(dataloader)
    # disp = ConfusionMatrixDisplay(confusion_matrix=CM,
    #                           display_labels=dataset.classes)

    print(f"Summary {mode.capitalize()} set")
    print(50*"=")
    print()
    print(f"""
{mode.lower()}_cost: {cost:.4f}\t
{mode.lower()}_acc: {acc:.4f}\t
{mode.lower()}_recall: {recall:.4f}\t
{mode.lower()}_precision: {precision:.4f}\t
{mode.lower()}_f1-score: {f1_:.4f}\t
    """)
    print()
    print(f"Focus on: {focus_on}")
    # print("Confusion Matrix")
    # disp.plot()
    # plt.show()

    if focus_on == "acc":
        score = acc
    elif focus_on == "recall":
        score = recall
    elif focus_on == "precision":
        score = precision
    elif focus_on == "f1":
        score = f1_


    if mode == 'test':
        return cost, score, CM
    return cost, score