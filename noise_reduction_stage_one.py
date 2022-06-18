from skimage.filters import threshold_otsu, threshold_multiotsu
import load_input
import numpy as np
from utilities import generate_probability_histograms, visualize_point_cloud
from math import sin
from math import cos

def initialize_global_constants():
    # Histogram information
    global hist_resolution
    hist_resolution = 312.5 * pow(10, -12)  # [s]
    global speed_of_light
    speed_of_light = 299792458  # [m/s]
    global num_probabilities_per_hist
    num_probabilities_per_hist = probabilities.shape[1] # info: only 1 of 12 represents the true distance
    global num_total_histograms
    num_total_histograms = probabilities.shape[0]
    global borders
    borders = np.array([107, 214, 321, 428, 535, 642, 749, 856, 963, 1070, 1177]).reshape(-1,1)

    print("Total number of histograms in input: ", num_total_histograms)


def tof_precision_limits():
    # dataset 1 has only true distances in its ground truth data, while, simulator generated datasets have dist,theta and phi data in its ground truth data
    if ground_truth_data.shape[1] > 1:  # check if the data belongs to simulator generated type
        distances_ground_truth = ground_truth_data["dist_true"]
    else:
        distances_ground_truth = ground_truth_data

    # Calculate time of flight from ground truth distances
    tof_ground_truth = (distances_ground_truth * 2) / speed_of_light

    # Min and Max values of tof precision.
    # Note: If the predicted tof is within +/- 5% of tof_ground_truth, it is considered as true prediction
    tof_min = 0.95 * tof_ground_truth
    tof_max = 1.05 * tof_ground_truth

    return tof_min, tof_max

def valid_hists_percentage(tof_min, tof_max):

    """
    :param tof_min:
    :param tof_max:
    :param hist_bins:
    :return: Percent_valid_hists # It is the percentage of valid histograms in the set of given histograms
    :Notes: Check validity of histograms - A histogram is valid if it contains the probability corresponding to true tof peak
    """

    valid_hists = np.zeros((num_total_histograms,1))
    num_true_points = 0

    for i in range(num_total_histograms):
        for j in range(num_probabilities_per_hist):
            if tof_min[i] < hist_bins.loc[i, j] * hist_resolution < tof_max[i]:
                valid_hists[i] = 1
                num_true_points += 1

    num_valid_hists = sum(valid_hists)
    percent_valid_hists = num_valid_hists * 100 / num_total_histograms

    print("Percentage of valid hists in input data: %.2f" % percent_valid_hists)

    return percent_valid_hists, num_true_points

def stage_one_analytics(percent_valid_hists, num_true_points_in_input, num_retentions, num_filtered_candidates):

    percent_true_points_before_filtering = (percent_valid_hists / (100 * num_probabilities_per_hist)) * 100
    percent_retentions_valid = (num_retentions / num_true_points_in_input) * 100  # Equal to percentage of valid true points retained
    percent_removed_candidates = 100 - (num_filtered_candidates / (num_total_histograms * num_probabilities_per_hist)) * 100
    percent_true_points_after_filtering = num_retentions * 100 / num_filtered_candidates

    print("Percentage of true points in unfiltered point cloud: %.2f" % percent_true_points_before_filtering)
    print("Percentage of points removed: %.2f" % percent_removed_candidates)
    print("Percentage of true points retained: %.2f" % percent_retentions_valid)
    print("Percentage of true points in filtered point cloud: %.2f" % percent_true_points_after_filtering)

def remove_border_effect():
    """
    This function eliminates border effect
    -> If a border has two candidates within 5% of its vicinity, it is highly likely that they belong to the same peak which falls on the border.
    -> If the above mentioned condition is satisfied at a border, we compare the probabilities of the two candidates and nullify the probability of the
        candidate whose probability is lower.
    -> The candidate whos probability is nullified will not be filtered by the next stage filter, which is OTSU based filter.

    """

    num_borders = num_probabilities_per_hist - 1
    next_border_skip_switch = 0
    for i in range(num_total_histograms):
        for j in range(num_borders):

            if next_border_skip_switch:
                next_border_skip_switch = 0
                continue

            #if hist_bins.loc[i,j] > borders[j,0]*0.95 and hist_bins.loc[i,j+1] < borders[j,0]*1.05:
            if hist_bins.loc[i, j] > borders[j, 0] - 16 and hist_bins.loc[i, j + 1] < borders[j, 0] + 16:
                new_prob = probabilities.loc[i, j] + probabilities.loc[i, j + 1]
                if probabilities.loc[i,j] > probabilities.loc[i,j+1]:

                    # Increase the probability of the current candidate
                    if new_prob < 1:
                        probabilities.loc[i, j] = new_prob
                    else:
                        probabilities.loc[i, j] = 1

                    # Nullify the probability of next candidate
                    probabilities.loc[i, j + 1] = 0
                else:
                    # Increase the probability of the next candidate
                    if new_prob < 1:
                        probabilities.loc[i, j+1] = new_prob
                    else:
                        probabilities.loc[i, j+1] = 1

                    # Nullify the probability of current candidate
                    probabilities.loc[i,j] = 0
                next_border_skip_switch = 1

    return probabilities

def otsu_filter(dataset = 2, sample_num = 1, object_dist = 12):

    # Initialize global variables
    global probabilities
    global hist_bins
    global ground_truth_data

    # Load Inputs
    probabilities, hist_bins, ground_truth_data = load_input.load(dataset, sample_num, min_dist = 6, object_dist= object_dist)

    # Initialize global constants
    initialize_global_constants()

    # Remove border effect
    probabilities = remove_border_effect()
    # Calculate tof limits
    tof_min, tof_max = tof_precision_limits()

    # Check the validity of histograms
    percent_valid_hists, num_true_points_in_input = valid_hists_percentage(tof_min, tof_max)

    '''
    # Apply OTSU on each histogram and extract the indices of filtered candidates
    '''
    # Initialise intermediate variables
    num_filtered_candidates = 0
    retentions = np.zeros((num_total_histograms, 1))
    otsu_bool_output = np.zeros((num_total_histograms, num_probabilities_per_hist))
    filtered_data_x_y_z_p_l = []

    bg_light = sample_num
    threshold_factor = -0.08*bg_light+0.7 # Best threshold factors for various BPRs are identified experimentally and this linear equation is fit

    for i in range(num_total_histograms):

        ### Thresholding the candidates of the histogram###
        # Calculate the threshold for histogram
        multi = 1
        nbins = 10
        if multi:
            try:
                thresholds = threshold_multiotsu(probabilities.loc[i], nbins=nbins)
                threshold = thresholds[0]
            except ValueError:
                threshold = threshold_otsu(probabilities.loc[i], nbins=nbins)*threshold_factor
        else:
            threshold = threshold_otsu(probabilities.loc[i], nbins=nbins)*0.2

        otsu_bool_output[i] = (probabilities.loc[i] > threshold).astype('int')
        num_filtered_candidates += np.sum(otsu_bool_output[i, :])

        ### Generating 3D coordinates and Labeling ###
        # Iterate through all the 12 candidates of the
        for j in range (num_probabilities_per_hist):

            if otsu_bool_output[i, j]==1:

                tof_pred = hist_bins.loc[i, j] * hist_resolution
                dist = tof_pred * speed_of_light/2

                if  dist>5.0:

                    # Avoid dataset 1, because it is 96k histograms dataset
                    if dataset != 1:
                        label = 0

                        # Label the candidate
                        if tof_min[i] < tof_pred < tof_max[i]:
                            retentions[i] = 1
                            label = 1

                        # Read angles of corresponding ground truth point
                        theta = ground_truth_data.loc[i, "theta"]
                        phi = ground_truth_data.loc[i, "phi"]

                        # Calculate x,y,z coordinates
                        x_value = dist * sin(theta) * cos(phi)
                        y_value = dist * sin(theta) * sin(phi)
                        z_value = dist * cos(theta)


                        # Append x,y,z,p,l values to the filtered data table
                        filtered_data_x_y_z_p_l.append([x_value, y_value, z_value, probabilities.loc[i,j], label])


    filtered_data_x_y_z_p_l = np.array(filtered_data_x_y_z_p_l)
    num_retentions = np.sum(filtered_data_x_y_z_p_l[:,4])
    print(filtered_data_x_y_z_p_l.shape)
    print("# OTSU filtering done #")

    # Analyse the filtered output
    #stage_one_analytics(percent_valid_hists, num_true_points_in_input, num_retentions , num_filtered_candidates)
    #generate_probability_histograms(filtered_data_x_y_z_p_l)

    # Visualize the output
    visualize_point_cloud(filtered_data_x_y_z_p_l)

    return filtered_data_x_y_z_p_l

def sota(dataset = 2, sample_num = 1):

    # Load the dataset if it is not loaded already.
    if not 'probabilities' in globals():

        global probabilities
        global hist_bins
        global ground_truth_data
        probabilities, hist_bins, ground_truth_data = load_input.load(dataset, sample_num)

    initialize_global_constants()

    # tof limits
    tof_min, tof_max = tof_precision_limits()

    # Check the validity of histograms
    percent_valid_hists, num_true_points_in_input = valid_hists_percentage(tof_min, tof_max)

    retentions = np.zeros((num_total_histograms, 1))

    # True prediction: If the predicted tof is within +/- 5% of tof_ground_truth, it is considered as true prediction
    for i in range(num_total_histograms):

        # Identify the index corresponding to the maximum probability
        max_probability_index = np.argmax(probabilities.loc[i])

        # Check if the identified candidate is the true detection
        if tof_min[i] < hist_bins.loc[i, max_probability_index] * hist_resolution < tof_max[i]:
            retentions[i] = 1
    print("# SOTA filtering done #")
    num_retentions = np.sum(retentions)
    num_filtered_candidates = num_total_histograms

    # Analyse the filtered output
    stage_one_analytics(percent_valid_hists, num_true_points_in_input,num_retentions, num_filtered_candidates)

