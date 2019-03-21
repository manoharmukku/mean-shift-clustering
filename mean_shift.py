"""
Author: Manohar Mukku
Date: 21 Mar 2019
Desc: Mean shift clustering algorithm
"""

import numpy as np

class MeanShift:
    def __init__(self, radius=1):
        self.radius = radius

    def fit(self, X):
        self.data = X
        self.centroids = X.copy()
        self.centroids.sort(axis=0)

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