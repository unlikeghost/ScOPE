import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import ConfusionMatrixDisplay


def plotConfMatrix(y_true:list, y_pred:list,
                   title:str, cmap:str='viridis',
                   labels:list=None, save:bool=False,):
    fig, ax = plt.subplots(figsize=(12,10))

    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true,
                                                   y_pred=y_pred,
                                                   ax=ax,
                                                   cmap=cmap,
                                                   display_labels=labels)
    
    disp.ax_.set_title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(f'{title}.png')
    plt.show()