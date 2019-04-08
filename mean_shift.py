"""
Author: Manohar Mukku
Date: 6 Apr 2019
Desc: Mean Shift Algorithm Implementation
"""

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


class MeanShift:
    def __init__(self, radius=2):
        self.radius = radius

    def fit(self, X):
        self.data = X
        self.centroids = []

        fig = plt.figure()
        
        max_plot_count = 4
        plot_count = 1

        # Iterate over all points individually
        for point in self.data:

            # Initially, point itself is the centroid
            centroid = point

            # Perform Mean Shift on this point until it converges
            while True:

                # List which stores all the points within the radius of current 'centroid'
                points_within_radius = []

                # Find all the points within the radius of 'centroid'
                for feature in self.data:
                    if (np.linalg.norm(feature-centroid) <= self.radius):
                        points_within_radius.append(feature)

                # Plot only for some points
                if (plot_count < max_plot_count):
                    # Plot the data
                    plt.scatter(self.data[:,0], self.data[:,1], c='blue', marker='o')

                    # Plot the centroid
                    plt.scatter(centroid[0], centroid[1], c='red', marker='+')

                    # Plot a circle around the centroid
                    circle = plt.Circle(centroid, self.radius, color='r', fill=False, clip_on=True)
                    plt.gca().add_artist(circle)

                    # Wait for a while and close the plot
                    plt.pause(1)
                    plt.close()

                # Save old centroid before it is updated
                old_centroid = centroid

                # Update centroid: New centroid is the mean of all the points within radius
                if (len(points_within_radius) > 0):
                    centroid = np.mean(points_within_radius, axis=0)

                # If centroid doesn't change, convergence reached, break
                if (np.array_equal(old_centroid, centroid)):
                    break

            plot_count += 1

            # Add the new found centroid to global centroids list
            self.centroids.append(centroid)

        # Remove duplicates from found centroids
        self.centroids = np.unique(self.centroids, axis=0)


if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Mean Shift Clustering')
    parser.add_argument('--radius', '-r', type=float, default=2, help='Radius/Bandwidth')
    parser.add_argument('--n_samples', '-n', type=int, default=500, help="No. of random samples to generate")
    args = parser.parse_args()

    # Generate data around 4 centers using make_blobs
    centers = [[-2,-2], [2,2], [2,-2], [-2,2]]
    
    X, _ = make_blobs(n_samples=args.n_samples, centers=centers, cluster_std=0.6)

    # Create mean_shift object
    ms = MeanShift(radius=args.radius)

    # Fit the data X
    ms.fit(X)

    # Get the final centroids
    centroids = ms.centroids

    print (centroids)

    # Plot the data and centroids
    plt.scatter(X[:,0], X[:,1], c='blue', marker='o')
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='+')
    for center in centroids:
        circle = plt.Circle(center, args.radius, color='r', fill=False, clip_on=True)
        plt.gca().add_artist(circle)
    plt.show()