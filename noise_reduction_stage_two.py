import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from utilities import generate_probability_histograms, visualize_point_cloud
"""
def iterative_guided_normal_filter():
"""

def find_threshold(filtered_data_x_y_z_p_l):
    # Dividing point cloud into regions and identifying the probability thresholds #
    region = 3
    min_dist = 0
    max_dist = 0
    if region == 1:
        min_dist = 0
        max_dist = 20

        threshold_regional = 0.8  # - 0.25 * sample_num

    elif region == 2:
        min_dist = 23.7
        max_dist = 26.3

        threshold_regional = 0.3  # - 0.25 * sample_num

    elif region == 3:
        min_dist = 47.4
        max_dist = 52.6

        threshold_regional = 0.14  # - 0.25 * sample_num

    # Visualize region and histograms of the region
    filtered_data_x_y_z_p_l = filtered_data_x_y_z_p_l[np.where(
        (filtered_data_x_y_z_p_l[:, 0] >= min_dist) & (filtered_data_x_y_z_p_l[:, 0] <= max_dist))]
    visualize_point_cloud(filtered_data_x_y_z_p_l)
    generate_probability_histograms(filtered_data_x_y_z_p_l, bins=100)

    # Filter the points based on the threshold
    filtered_data_x_y_z_p_l_new = filtered_data_x_y_z_p_l[np.where(filtered_data_x_y_z_p_l[:, 3] > threshold_regional)]
    visualize_point_cloud(filtered_data_x_y_z_p_l_new)
    #generate_probability_histograms(filtered_data_x_y_z_p_l_new)

    # Calculate the percentage of true points retained
    num_true_points = 0
    count = 0
    for i in range(filtered_data_x_y_z_p_l.shape[0]):
        if filtered_data_x_y_z_p_l[i, 4] == 1:
            num_true_points += 1
            if filtered_data_x_y_z_p_l[i, 3] > threshold_regional:
                count += 1
    percentage = (count / num_true_points) * 100

    print(count)
    print(percentage)

def bilateral_filter_3d(point_cloud_data):
    filtered_data_x_y_z_p_l = point_cloud_data

    pcd_otsu_result = o3d.geometry.PointCloud()
    pcd_otsu_result.points = o3d.utility.Vector3dVector(filtered_data_x_y_z_p_l[:, 0:3])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_otsu_result)
    coordinates = filtered_data_x_y_z_p_l[:, 0:3]
    var_dist = 2.5

    for i in range(2):
        # Initialize containers to store updated coordinates
        updated_x_coordinates = np.zeros((filtered_data_x_y_z_p_l.shape[0], 1))
        updated_y_coordinates = np.zeros((filtered_data_x_y_z_p_l.shape[0], 1))
        updated_z_coordinates = np.zeros((filtered_data_x_y_z_p_l.shape[0], 1))

        # For each point in the point cloud
        for n in range(filtered_data_x_y_z_p_l.shape[0]):

            # Extract the indices of k nearest neighbours of the point in question
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd_otsu_result.points[n], 0.2)

            weights = []

            # Extract and store the coordinates of the neighbours
            x_coordinates_of_neighbours = np.take(filtered_data_x_y_z_p_l[:, 0], idx, axis=0)
            y_coordinates_of_neighbours = np.take(filtered_data_x_y_z_p_l[:, 1], idx, axis=0)
            z_coordinates_of_neighbours = np.take(filtered_data_x_y_z_p_l[:, 2], idx, axis=0)

            # Calculate weights for the neighbours
            for id_n in idx:
                term = (np.linalg.norm(coordinates[id_n] - coordinates[n]) ** 2) / (2 * var_dist)
                weight = np.exp(-term)
                weights.append(weight)

            # Update the p value of point of interest
            updated_x_coordinates[n, 0] = sum(x_coordinates_of_neighbours[i] * weights[i] for i in range(len(idx))) / sum(weights)
            updated_y_coordinates[n, 0] = sum(y_coordinates_of_neighbours[i] * weights[i] for i in range(len(idx))) / sum(weights)
            updated_z_coordinates[n, 0] = sum(z_coordinates_of_neighbours[i] * weights[i] for i in range(len(idx))) / sum(weights)


        # Replace the probabilities in the data table with the updated probabilities
        filtered_data_x_y_z_p_l[:, 0] = updated_x_coordinates[:, 0]
        filtered_data_x_y_z_p_l[:, 1] = updated_y_coordinates[:, 0]
        filtered_data_x_y_z_p_l[:, 2] = updated_z_coordinates[:, 0]

    # Visualize point cloud that is the result of bilateral filtering
    visualize_point_cloud(filtered_data_x_y_z_p_l)

    return filtered_data_x_y_z_p_l
def visualize_scatter_plots(filtered_data_x_y_z_p_l):
    # True points scatter plot
    filtered_data_x_y_z_p_l_true = filtered_data_x_y_z_p_l[np.where(filtered_data_x_y_z_p_l[:, 4] == 1)]
    all_xs = filtered_data_x_y_z_p_l_true[:, 0]
    all_ps = filtered_data_x_y_z_p_l_true[:, 3]
    marker_size = 0.001
    plt.scatter(all_xs, all_ps, s=marker_size)
    #plt.show()

    # False points scatter plot
    filtered_data_x_y_z_p_l_false = filtered_data_x_y_z_p_l[np.where(filtered_data_x_y_z_p_l[:, 4] == 0)]
    all_xs = filtered_data_x_y_z_p_l_false[:, 0]
    all_ps = filtered_data_x_y_z_p_l_false[:, 3]

    plt.scatter(all_xs, all_ps, s=marker_size)
    plt.show()

def line_equation(points):
    m = (points[1,1] - points[0,1])/(points[1,0] - points[0,0])
    c = points[0,1] - m*points[0,0]
    return m,c

def dynamic_probability_threshold_filter(filtered_data_x_y_z_p_l):

    # <write comments>
    threshold_limits = np.array([((5, 0.8), (53, 0.4)),
                                 ((5, 0.8), (53, 0.4)),
                                 ((5, 0.8), (53, 0.32)),
                                 ((5, 0.8), (53, 0.3)),
                                 ((5, 0.8), (53, 0.23)),
                                 ((5, 0.8), (53, 0.2)),
                                 ((5, 0.8), (53, 0.16)),
                                 ((5, 0.8), (53, 0.14))
                                 ])

    threshold_m, threshold_c = line_equation(threshold_limits[BPR-1])

    probability_based_adaptive_filtered_data_x_y_z_p_l = []
    for i in range(filtered_data_x_y_z_p_l.shape[0]):

        probability_threshold = threshold_m * filtered_data_x_y_z_p_l[i, 0] + threshold_c

        if filtered_data_x_y_z_p_l[i, 3] > probability_threshold:  # Check probability
            probability_based_adaptive_filtered_data_x_y_z_p_l.append(filtered_data_x_y_z_p_l[i, :])

    probability_based_adaptive_filtered_data_x_y_z_p_l = np.array(probability_based_adaptive_filtered_data_x_y_z_p_l)
    return probability_based_adaptive_filtered_data_x_y_z_p_l

def region_based_probability_threshold_filter(filtered_data_x_y_z_p_l):
    region_1_threshold = 0.8
    region_2_threshold = 0.4
    region_3_threshold = 0.25
    region_4_threshold = 0.20

    region_1_border = 30
    region_2_border = 40
    region_3_border = 50


    region_based_probability_threshold_filtered_data_x_y_z_p_l = []

    for i in range(filtered_data_x_y_z_p_l.shape[0]):

        # Find respective probability threshold
        if filtered_data_x_y_z_p_l[i,0] < region_1_border:
            probability_threshold = region_1_threshold
        elif filtered_data_x_y_z_p_l[i,0] < region_2_border:
            probability_threshold = region_2_threshold
        elif filtered_data_x_y_z_p_l[i,0] < region_3_border:
            probability_threshold = region_3_threshold
        else:
            probability_threshold = region_4_threshold

        # Apply the threshold
        if filtered_data_x_y_z_p_l[i, 3] > probability_threshold:  # Check probability
            region_based_probability_threshold_filtered_data_x_y_z_p_l.append(filtered_data_x_y_z_p_l[i, :])

    probability_based_adaptive_filtered_data_x_y_z_p_l = np.array(region_based_probability_threshold_filtered_data_x_y_z_p_l)
    return probability_based_adaptive_filtered_data_x_y_z_p_l

def probability_threshold_filter(filtered_data_x_y_z_p_l):
    # <write comments>
    threshold_limits = np.array([((40, 0.78), (50, 0.76)),
                                 ((40, 0.75), (50, 0.70)),
                                 ((40, 0.57), (50, 0.52)),
                                 ((40, 0.40), (50, 0.35)),
                                 ((40, 0.35), (50, 0.27)),
                                 ((40, 0.30), (50, 0.23)),
                                 ((40, 0.25), (50, 0.20)),
                                 ((40, 0.20), (50, 0.18)),
                                 ])

    probability_threshold_filtered_data_x_y_z_p_l = []

    apply_global_threshold = False

    #
    if BPR <= 2:
        apply_global_threshold = True
        global_reshold = 0.8
    else:
        # region_1 = [0,30m] along x-axis
        threshold_region_1 = 0.8

        # region_2 = [30,40m] along x-axis
        threshold_region_2_m, threshold_region_2_c = line_equation(np.array([(30,threshold_region_1),
                                                                             threshold_limits[BPR-1,0]])
                                                                   )
        # region_2 = [40,60] along x-axis
        threshold_region_3_m, threshold_region_3_c = line_equation(threshold_limits[BPR - 1])

    # Filtering
    for i in range(filtered_data_x_y_z_p_l.shape[0]):

        if apply_global_threshold:
            probability_threshold = global_reshold
        else:
            # Find respective probability threshold
            if filtered_data_x_y_z_p_l[i, 0] < 30:
                probability_threshold = threshold_region_1
            elif filtered_data_x_y_z_p_l[i, 0] < 40:
                probability_threshold = threshold_region_2_m * filtered_data_x_y_z_p_l[i, 0] + threshold_region_2_c
            else:
                probability_threshold = threshold_region_3_m * filtered_data_x_y_z_p_l[i, 0] + threshold_region_3_c

        # Apply the threshold
        if filtered_data_x_y_z_p_l[i, 3] > probability_threshold:  # Check probability
            probability_threshold_filtered_data_x_y_z_p_l.append(filtered_data_x_y_z_p_l[i, :])

    probability_based_adaptive_filtered_data_x_y_z_p_l = np.array(probability_threshold_filtered_data_x_y_z_p_l)
    return probability_based_adaptive_filtered_data_x_y_z_p_l

def dynamic_radius_outlier_removal():
    return 0

def bilateral_filter(filtered_data_from_stage_one_filtering, sample_num = 1):

    global BPR
    BPR = sample_num
    filtered_data_x_y_z_p_l = filtered_data_from_stage_one_filtering

    pcd_otsu_result = o3d.geometry.PointCloud()
    pcd_otsu_result.points = o3d.utility.Vector3dVector(filtered_data_x_y_z_p_l[:, 0:3])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_otsu_result)
    coordinates = filtered_data_x_y_z_p_l[:, 0:3]

    # Variance Limits: defining minimum and maximum limits of the variance for bi-lateral filter. Using these limits to create a linear equation.
    #                  This line equation is then used to calculate the variance for each point with respect to its x-coordinate
    points = np.array([(5,0.15),(60,1.5)]) # X-Coordinates : distance; Y-Coordinates : Variance
    var_m, var_c = line_equation(points)

    for i in range(2):
        updated_probabilities = np.zeros((filtered_data_x_y_z_p_l.shape[0],1))
        for n in range(filtered_data_x_y_z_p_l.shape[0]):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd_otsu_result.points[n], 10)
            weights = []
            probabilities_of_neighbours = np.take(filtered_data_x_y_z_p_l[:, 3], idx, axis=0)

            # Calculate weights for the neighbours
            for id_n in idx:
                var_dist = var_m*coordinates[i,0]+var_c # Returns variance with respect to the x coordiante of the point in the 3d Space
                term = (np.linalg.norm(coordinates[id_n] - coordinates[n]) ** 2) / (2 * var_dist)
                weight = np.exp(-term)
                weights.append(weight)

            # Update the p value of point of interest
            updated_probabilities[n, 0] = sum(probabilities_of_neighbours[i] * weights[i] for i in range(len(idx))) / sum(weights)

        # Replace the probabilities in the data table with the updated probabilities
        filtered_data_x_y_z_p_l[:,3] = updated_probabilities[:,0]

    # Visualize point cloud that is the result of bilateral filtering
    visualize_point_cloud(filtered_data_x_y_z_p_l)

    visualize_scatter_plots(filtered_data_x_y_z_p_l)

    # Visualize histograms of points of the point cloud that is the result of bilateral filtering
    #generate_probability_histograms(filtered_data_x_y_z_p_l)

    """
    Probability based filtering
    """
    #adaptive_probability_based_filtered_data_x_y_z_p_l = dynamic_probability_threshold_filter(filtered_data_x_y_z_p_l)
    adaptive_probability_based_filtered_data_x_y_z_p_l = probability_threshold_filter(filtered_data_x_y_z_p_l)
    visualize_point_cloud(adaptive_probability_based_filtered_data_x_y_z_p_l)

    """
    find_threshold(filtered_data_x_y_z_p_l)
    """

    # Statistical Outlier removal #
    print("Statistical oulier removal")
    pcd_bilateral_result = o3d.geometry.PointCloud()
    pcd_bilateral_result.points = o3d.utility.Vector3dVector(adaptive_probability_based_filtered_data_x_y_z_p_l[:, 0:3])
    cl, ind = pcd_bilateral_result.remove_statistical_outlier(nb_neighbors=10,
                                                              std_ratio=1.0)
    # display_inlier_outlier(pcd_otsu_result, ind)
    pcd_filtered_sor = np.take(adaptive_probability_based_filtered_data_x_y_z_p_l, ind, axis=0)
    print(pcd_filtered_sor.shape)
    visualize_point_cloud(pcd_filtered_sor)
    visualize_scatter_plots(pcd_filtered_sor)
    #filtered_data_x_y_z_p_l = bilateral_filter_3d(pcd_filtered_sor)

    # Apply bilateral filter on the 3d coordinates
    filtered_data_x_y_z_p_l = bilateral_filter_3d(pcd_filtered_sor)