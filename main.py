"""
Author: Manohar Mukku
Date: 21 Mar 2019
Desc: Mean shift clustering algorithm
"""

from sklearn.datasets.samples_generator import make_blobs
import argparse
from mean_shift import *

if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Mean Shift Clustering')
    parser.add_argument('--radius', '-r', type=int, default=1, help='Radius/Bandwidth')
    parser.add_argument('--n_samples', '-n', type=int, default=500, help="No. of random samples to generate")
    args = parser.parse_args()

    # Generate data around 4 centers using make_blobs
    centers = [[-2,-2], [2,2], [2,-2], [-2,2]]
    
    X, _ = make_blobs(n_samples=args.n_samples, centers=centers, cluster_std=1)

    # Create mean_shift object
    ms = MeanShift(radius=args.radius)

    # Fit the data X
    ms.fit(X)

    # Find the final centroids
    centroids = ms.centroids

    print (centroids)