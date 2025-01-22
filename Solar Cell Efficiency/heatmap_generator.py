import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def generate_topographic_heatmap(width, height, num_clusters=5):
    '''
    Given a desired width, height, and number of clusters, create 
    topographic heatmap data with those given filters.
    Return the data in an array.
    '''
    cluster_centers = np.random.rand(num_clusters, 2) * [width, height]
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    heatmap_data = np.zeros((height, width))
    for x, y in cluster_centers:
        distance = np.sqrt((X - x)**2 + (Y - y)**2)
        heatmap_data += np.exp(-distance / 20)
    heatmap_data_smoothed = gaussian_filter(heatmap_data, sigma=3)

    return heatmap_data_smoothed

def plot_heatmap(heatmap_data, filename):
    '''
    Given heatmap data, create a heatmap.
    Save it to the project folder with the given filename.
    '''
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar().remove()
    plt.axis('off')
    folder_path = "Solar_Cell_Images"
    file_name = filename
    plt.savefig(os.path.join(folder_path, file_name), bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.close

def compute_gradient(data):
    '''
    Given heatmap data, find the gradient of the data (derivate).
    Return the data in an array.
    '''
    dx = np.gradient(data, axis=1)
    dy = np.gradient(data, axis=0)

    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    return gradient_magnitude


width = 200
height = 200  
num_clusters = 20


heatmap_data = generate_topographic_heatmap(width, height, num_clusters)
gradient_data = compute_gradient(heatmap_data)

plot_heatmap(heatmap_data, "heatmap.jpg")
plot_heatmap(gradient_data, "gradientmap.jpg")

          
