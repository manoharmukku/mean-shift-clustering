"""
Author: Manohar Mukku
Date: 21 Mar 2019
Desc: Mean shift clustering algorithm
"""

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

class MeanShift:
    def __init__(self, radius=1):
        self.radius = radius

    def fit(self, X):
        self.data = X.copy()
        self.centroids = X.copy()
        self.centroids.sort(axis=0)

        fig = plt.figure()

        while True:

            # List which stores all the new centroids formed
            new_centroids = []

            # Iterate over all centroids individually
            for centroid in self.centroids:

                # List which stores all the points within the radius of current 'centroid'
                points_within_radius = []

                # Find all the points within the radius of 'centroid'
                for point in self.data:
                    if (np.linalg.norm(point-centroid) <= self.radius):
                        points_within_radius.append(point)

                if (len(points_within_radius) > 0):
                    points_within_radius = np.array(points_within_radius)

                    new_centroid = np.mean(points_within_radius, axis=0)

                    new_centroids.append(new_centroid)

            # Remove duplicates
            new_centroids = np.unique(new_centroids, axis=0)

            # Sort new_centroids
            new_centroids.sort(axis=0)

            # If new_centroids is same as old centroids, saturation reached, stop
            if (np.array_equal(self.centroids, new_centroids)):
                break

            # Else, update the old centroids with the new ones
            self.centroids = new_centroids.copy()

            # Plot the data
            plt.scatter(self.data[:,0], self.data[:,1], c='blue', marker='o')
            plt.scatter(self.centroids[:,0], self.centroids[:,1], c='red', marker='+')
            plt.pause(1)
            plt.close()


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

    # Plot the data
    plt.scatter(X[:,0], X[:,1], c='blue', marker='o')
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='+')
    plt.show()