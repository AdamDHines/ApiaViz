import numpy as np

import torch
import matplotlib.pyplot as plt
from apiaviz.src.metrics import recallAtK

def run_sad(GTtol, query, reference):

    # Load and preprocess images from both folders

    # Track progress for both folders
    images1 = query
    images2 = reference

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = images1[:, 1:3, :, :].reshape(images1.shape[0], -1).unsqueeze(0).to(device, dtype=torch.float32)

    # Do the same for the second set of images
    b = images2[:, 1:3, :, :].reshape(images2.shape[0], -1).unsqueeze(0).to(device, dtype=torch.float32)

    # Track progress for calculating distance
    torch_dist = torch.cdist(b, a, 1)[0]

    # Perform sequence matching convolution on similarity matrix
    dist_matrix_seq = torch_dist.cpu().numpy()

    # save distance matrix as a pdf image
    plt.imshow(dist_matrix_seq)
    plt.colorbar()
    plt.show()
    plt.close()

    N = [1,5,10,15,20,25] # N values to calculate
    

    # Calculate Recall@N
    recallatn = []
    for n in N:
        recallatn.append(round(recallAtK(1/dist_matrix_seq,GTtol,K=n),2))


    return recallatn