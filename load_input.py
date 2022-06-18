import pandas as pd
import os

def dataset1(hist_bins_path, prediction_data_path):
    """
    :param hist_bins_path:
    :param prediction_data_path:
    :return: probabilities, hist_bins, distances_ground_truth
    """

    hist_bins = pd.read_csv(hist_bins_path)
    hist_bins = pd.DataFrame(hist_bins.bins.str.split(' ', expand=True))
    hist_bins = hist_bins.apply(pd.to_numeric, errors='coerce')

    prediction_data = pd.read_csv(prediction_data_path)
    prediction_data = pd.DataFrame(prediction_data.predictions.str.split('   ', expand=True))
    prediction_data = prediction_data.apply(pd.to_numeric, errors='coerce')
    prediction_data = prediction_data.rename(
        {0: 'p1', 1: 'p2', 2: 'p3', 3: 'p4', 4: 'p5', 5: 'p6', 6: 'p7', 7: 'p8', 8: 'p9', 9: 'p10', 10: 'p11',
         11: 'p12',
         12: 'target_dist', 13: 'bg_intensity'}, axis='columns')

    '''
    #Extract data into usable form
    '''
    # Create separate tables for probabilities, ground truth distances and back-ground light intensities
    # The data is in "object" datatype by default. "pd.to_numeric" is used to convert data to float datatype
    probabilities = prediction_data.loc[:, 'p1':'p12']  # range of probabilities: [0,1]
    probabilities = probabilities.apply(pd.to_numeric, errors='coerce')
    probabilities = probabilities
    distances_ground_truth = pd.to_numeric(prediction_data.loc[:, 'target_dist'], errors='coerce');  # [Meter]

    dataset1_indices = prediction_data.index[
        (prediction_data['target_dist'] >= 0.0) & (prediction_data['target_dist'] <= 60.0) & (
                    prediction_data['bg_intensity'] == 8000000.0)]

    print("Size of the dataset: ", dataset1_indices.size)

    # Final step in data extraction
    # When a subset of a dataframe is created, it maintains old row numbers. Hence we reset them
    probabilities = probabilities.iloc[dataset1_indices, :]
    hist_bins = hist_bins.iloc[dataset1_indices, :]
    distances_ground_truth = distances_ground_truth.iloc[dataset1_indices]

    probabilities.reset_index(drop=True, inplace=True)
    hist_bins.reset_index(drop=True, inplace=True)
    distances_ground_truth.reset_index(drop=True, inplace=True)
    # Note: dataset 1 has only distances information about the ground truth, but not the angles information

    return probabilities, hist_bins, distances_ground_truth


def dataset2(probabilities_path, hist_bins_path, ground_truth_data_path):

    probabilities = pd.read_csv(probabilities_path, header=None)
    probabilities.rename({0: 'probabilities'}, axis='columns', inplace=True)
    probabilities = probabilities.probabilities.str.split(' ', expand=True)
    probabilities = probabilities.apply(pd.to_numeric, errors='coerce')

    hist_bins = pd.read_csv(hist_bins_path, header=None)
    hist_bins.rename({0: 'hist_indices'}, axis='columns', inplace=True)
    hist_bins = hist_bins.hist_indices.str.split(' ', expand=True)
    hist_bins = hist_bins.apply(pd.to_numeric, errors='coerce')

    #load theta and phi angles of ground truth data
    ground_truth_data = pd.read_csv(ground_truth_data_path, header = None)
    ground_truth_data.rename({0: 'ground_truth_information'}, axis='columns', inplace=True)
    ground_truth_data = ground_truth_data.ground_truth_information.str.split(' ', expand=True)
    ground_truth_data = ground_truth_data.apply(pd.to_numeric, errors='coerce')
    ground_truth_data = ground_truth_data.loc[:, 3:5]
    ground_truth_data.rename({3: 'dist_true', 4:'theta', 5:'phi'}, axis='columns', inplace=True)

    return probabilities, hist_bins, ground_truth_data

def remove_points_below_min_dist(probabilities, hist_bins, ground_truth_data, min_dist):
    dataset1_indices = ground_truth_data.index[
        (ground_truth_data['dist_true'] >= min_dist)]

    #print("Size of the dataset: ", dataset1_indices.size)

    # Final step in data extraction
    # When a subset of a dataframe is created, it maintains old row numbers. Hence we reset them
    probabilities = probabilities.iloc[dataset1_indices, :]
    hist_bins = hist_bins.iloc[dataset1_indices, :]
    ground_truth_data = ground_truth_data.iloc[dataset1_indices]

    probabilities.reset_index(drop=True, inplace=True)
    hist_bins.reset_index(drop=True, inplace=True)
    ground_truth_data.reset_index(drop=True, inplace=True)

    return probabilities, hist_bins, ground_truth_data

def load(dataset, sample_num = 1, min_dist = 5.0, object_dist = 12):

    if dataset == 1:
        # dataset I
        hist_bins_path = "C:\ML for LiDAR point clouds\Development\My developement\EDA on histograms\Dataset\Prediction_FNN\indices_all.txt"
        prediction_data_path = "C:\ML for LiDAR point clouds\Development\My developement\EDA on histograms\Dataset\Prediction_FNN\p_NN_all.txt"
        probabilities, hist_bins, ground_truth_data = dataset1(hist_bins_path, prediction_data_path)
    elif dataset == 2:
        # dataset II
        sample_folder_name = str(sample_num) + "mhz"
        sample_folder_path= r'C:\ML for LiDAR point clouds\Development\My developement\Point cloud processing\data\car_25meters_8000_points'
        probabilities_path = os.path.join(sample_folder_path, sample_folder_name,r'probabilities.txt')
        hist_bins_path = os.path.join(sample_folder_path, sample_folder_name,r'indices.txt')
        ground_truth_data_path = os.path.join(sample_folder_path, sample_folder_name,r'x_true_y_true_z_true_dist_theta_phi.txt')
        probabilities, hist_bins, ground_truth_data = dataset2(probabilities_path, hist_bins_path, ground_truth_data_path)
    elif dataset == 3:
        # dataset III
        sample_folder_name = str(sample_num) + "mhz"
        sample_folder_path= r'C:\ML for LiDAR point clouds\Development\My developement\Point cloud processing\data\car_38meters_8000_points'
        probabilities_path = os.path.join(sample_folder_path, sample_folder_name,r'probabilities.txt')
        hist_bins_path = os.path.join(sample_folder_path, sample_folder_name,r'indices.txt')
        ground_truth_data_path = os.path.join(sample_folder_path, sample_folder_name,r'x_true_y_true_z_true_dist_theta_phi.txt')
        probabilities, hist_bins, ground_truth_data = dataset2(probabilities_path, hist_bins_path, ground_truth_data_path)
    elif dataset == 4:
        # dataset III
        sample_folders_folder_name = str(object_dist) + "mts"
        sample_folder_name = str(sample_num) + "mhz"
        sample_folder_path = r'C:\ML for LiDAR point clouds\Development\My developement\Point cloud processing\data\car at equal distance intervals'
        probabilities_path = os.path.join(sample_folder_path, sample_folders_folder_name ,sample_folder_name, r'probabilities.txt')
        hist_bins_path = os.path.join(sample_folder_path, sample_folders_folder_name ,sample_folder_name, r'indices.txt')
        ground_truth_data_path = os.path.join(sample_folder_path, sample_folders_folder_name ,sample_folder_name,r'x_true_y_true_z_true_dist_theta_phi.txt')
        probabilities, hist_bins, ground_truth_data = dataset2(probabilities_path, hist_bins_path,ground_truth_data_path)
    else:
        # Indivisual carla dataset
        probabilities_path = r'C:\ML for LiDAR point clouds\Development\My developement\Point Cloud Simulator\data\lidar_data\probabilities.txt'
        hist_bins_path = r'C:\ML for LiDAR point clouds\Development\My developement\Point Cloud Simulator\data\lidar_data\indices.txt'
        ground_truth_data_path= r'C:\ML for LiDAR point clouds\Development\My developement\Point Cloud Simulator\data\lidar_data\2_Distanz_Winkel_Berechnung\x_true_y_true_z_true_dist_theta_phi\x_true_y_true_z_true_dist_theta_phi.txt'
        probabilities, hist_bins, ground_truth_data = dataset2(probabilities_path, hist_bins_path,ground_truth_data_path)

    probabilities, hist_bins, ground_truth_data = remove_points_below_min_dist(probabilities, hist_bins, ground_truth_data, min_dist)
    return probabilities, hist_bins, ground_truth_data