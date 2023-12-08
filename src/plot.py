import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_image_grid(images, labels=None, ncols=10, figsize=(5,5), title="", cmap=None, channels_first=True):
    """
    This function will plot given images in a rectangular grid.
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    images = np.moveaxis(images, 1,3) if channels_first==True and len(images.shape)==4 else images
    images = images.squeeze()
    N, H, W = images.shape[:3]
    nrows = int(np.ceil(N / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True)
    ax = np.ravel(ax)
    cmap = cmap if len(images.shape)==4 else "gray"
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap=cmap)
        if labels is not None and len(labels)>0:
            ax[i].set_title(labels[i])
    fig.suptitle(title)
    fig.show()

