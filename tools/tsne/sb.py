import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
%matplotlib inline

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123
# need to create a subset of the data, too much time to process otherwise
# features['train': 800 * 128, 'test':200 * 128 ]  
x_subset = features['train'][:5000]
y_subset = targets['train'][:5000]


labels = {
     0: 'blues',  
     1: 'classical',
     2: 'country',
     3: 'disco',
     4: 'hippop',
     5: 'jazz',
     6: 'metal',
     7: 'pop',
     8: 'reggae',
     9: 'rock',
}

#visulize
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

   
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(labels[i]), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts