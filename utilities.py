import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

def generate_probability_histograms(data, bins = 200):
    probabilities_of_filtered_candidates = data[:, 3]
    labels = (data[:, 4]).reshape(-1, 1)
    print(labels.shape)

    # Accumulate the probabilities of true and false candidates separately
    probabilities_of_true_candidates = []
    probabilities_of_false_candidates = []
    for i in range(data.shape[0]):
            if labels[i] == 1:
                probabilities_of_true_candidates.append(probabilities_of_filtered_candidates[i])
            else:
                probabilities_of_false_candidates.append(probabilities_of_filtered_candidates[i])

    # Plot histograms of probabilities of all candidates, true candidates, false candidates
    plt.hist(probabilities_of_filtered_candidates, bins=bins)
    plt.title("All points")
    plt.show()
    plt.hist(probabilities_of_true_candidates, bins=bins)
    plt.title("True points")
    plt.show()
    plt.hist(probabilities_of_false_candidates, bins=bins)
    plt.title("False points")
    plt.show()

def visualize_point_cloud(filtered_data_x_y_z_p_l):

    # Visualize point cloud #
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_data_x_y_z_p_l[:, 0:3])
    ## Assign color to each point with respect to its probability value: Black-0.0; Red:1.0
    probabilities_of_filtered_candidates = filtered_data_x_y_z_p_l[:, 3]
    probabilities_of_filtered_candidates = probabilities_of_filtered_candidates.reshape(-1,1)
    colors = np.concatenate((probabilities_of_filtered_candidates,
                             np.zeros((len(probabilities_of_filtered_candidates), 1)),
                             np.zeros((len(probabilities_of_filtered_candidates), 1))), axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    ## Visualization
    o3d.visualization.draw_geometries([pcd])