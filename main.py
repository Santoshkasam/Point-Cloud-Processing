import noise_reduction_stage_one
import noise_reduction_stage_two
import load_input

dataset = 2 # 1: 96k dataset, 2: saved carla dataset, 3: current simulation output from point cloud simulator

object_dist = 42
sample_num = 7 # number represents background photon rate in mhz
print("Sample:", sample_num, "Mhz")

filtered_data_x_y_z_p_l = noise_reduction_stage_one.otsu_filter(dataset, sample_num, object_dist = object_dist)
#noise_reduction_stage_one.sota(dataset, sample_num)
noise_reduction_stage_two.bilateral_filter(filtered_data_x_y_z_p_l, sample_num)

