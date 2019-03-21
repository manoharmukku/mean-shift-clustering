"""
Author: Manohar Mukku
Date: 21 Mar 2019
Desc: Mean shift clustering algorithm
"""

import numpy as np

class MeanShift:
    def __init__(self, radius=1):
        self.radius = radius

    def fit(X):
        self.data = X
        self.centroids = sorted(X)

        while True:
            new_centroids = []

            for center in centroids:
                within_radius = []

                for point in data:
                    if (np.linalg.norm(point-center) <= self.radius):
                        within_radius.append(point)

                new_centroid = np.average(within_radius)

                new_centroids.append(new_centroid)

            # Remove duplicates
            new_centroids = sorted(list(set(new_centroids)))

            # Compare new_centroids with old centroids
            if (new_centroids == centroids):
                break

            centroids = new_centroids.copy()