
#_________________
#
# CONFUSION MATRIX
#_________________
#

import matplotlib.pyplot as plt
from sklearn import metrics as sm
import numpy as np

def plot_cm(predictions, actuals, classes, normalize, cmap, figsz, title):
    
    from itertools import product
    
    cm = sm.confusion_matrix(predictions, actuals, labels=[i for i in range(len(classes))])
    
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.rcParams["figure.figsize"] = figsz
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes, rotation=25)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()
    
    return i, j

#__________________________________________________
#
# Area Under the Receiver Operating Characteristics
#__________________________________________________
#

import matplotlib.pyplot as plt
from sklearn import metrics as sm
from itertools import cycle
from scipy import interp
import numpy as np
  
def plot_auroc(y_pred, y_test, classes, title):
        
    lw = 2
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = sm.roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = sm.auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = sm.roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = sm.auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)): mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= len(classes)
    
    fpr["macro"] = all_fpr; tpr["macro"] = mean_tpr
    roc_auc["macro"] = sm.auc(fpr["macro"], tpr["macro"])
    
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
#______________________
#
# CLASSIFICATION REPORT
#______________________
#

import matplotlib.pyplot as plt
from sklearn import metrics as sm
import numpy as np

def show_values(pc, fmt="%.2f", **kw):
    
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation=False, cmap='RdBu'):
    
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      
    plt.xlim( (0, AUC.shape[1]) )

    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.colorbar(c)
    show_values(c)

    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))

def plot_clr(predictions, actuals, classes, cmap, figsz, title):
    
    classes = [x.replace(" ", "_") for x in classes]
    
    cr = sm.classification_report(actuals, predictions, labels=[i for i in range(len(classes))], target_names=classes)

    lines = cr.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().replace(' avg', '-avg').split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        #print(v)
        plotMat.append(v)

    #print('plotMat: {0}'.format(plotMat))
    #print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    correct_orientation = False
    
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figsz[0], figsz[1], correct_orientation, cmap=cmap)

#________________
#
# DERIVED OUTPUTS
#________________
#
    
import matplotlib.pyplot as plt
from sklearn import metrics as sm

def output_derivations(predictions, actuals, y_pred, y_test, classes, name):
        
    plot_cm(predictions, actuals, classes, normalize=True, cmap=plt.cm.BuPu, figsz=(12,12), title="Confusion Matrix")
    plt.savefig("diagnostic_cm.png", dpi=200, format='png', bbox_inches='tight', pad_inches=0.5); plt.close();
    
    plot_auroc(y_pred, y_test, classes, title = 'Area Under the Reciever Operating Characteristics')
    plt.savefig("diagnostic_auroc.png", dpi=200, format='png', bbox_inches='tight', pad_inches=0.5); plt.close();
    
    plot_clr(predictions, actuals, classes, cmap='RdBu', figsz=(30,15), title = 'Classification Report')
    plt.savefig("diagnostic_clr.png", dpi=200, format='png', bbox_inches='tight', pad_inches=0.25); plt.close();
    
    fig = plt.figure(figsize=(9, 10))
    
    acc = str(round(sm.accuracy_score(predictions, actuals)*100, 3))
    kappa = str(round(sm.cohen_kappa_score(predictions, actuals), 3))
    
    fig.suptitle("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n" + "*** ACCURACY = "+acc+"% | COHEN'S KAPPA = "+kappa+" ***", fontsize=17.5, fontweight="bold")
    
    fig.add_subplot(221); plt.imshow(plt.imread("diagnostic_cm.png")); plt.axis('off'); os.remove("diagnostic_cm.png")
    fig.add_subplot(222); plt.imshow(plt.imread("diagnostic_auroc.png")); plt.axis('off'); os.remove("diagnostic_auroc.png")
    fig.add_subplot(212); plt.imshow(plt.imread("diagnostic_clr.png")); plt.axis('off'); os.remove("diagnostic_clr.png")
    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    
    plt.savefig("diagnostic_output_derivations.png", dpi=700, format='png'); plt.close();