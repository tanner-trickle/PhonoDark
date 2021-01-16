import h5py
import numpy as np

import src.physics as physics

def hdf5_write_dict(hdf5_file, group_name, data_dict):
    """
        Recursively go through a dictionary and write all values to the group_name
    """

    for index in data_dict:

        if type(data_dict[index]) is dict:

            hdf5_write_dict(hdf5_file, group_name+'/'+str(index), data_dict[index])

        else:

            hdf5_file.create_dataset(group_name+'/'+str(index), data=data_dict[index])

def hdf5_write_output(out_file, numerics_parameters, physics_parameters, dm_properties_dict, c_dict,
                        all_total_rate_list, n_proc, material, all_diff_rate_list,
                        all_binned_rate_list):
    """
        
        Write data to hdf5 file

    """

    out_f = h5py.File(out_file, "w")

    out_f.create_group('numerics')
    out_f.create_group('particle_physics')
    out_f.create_group('material_properties')
    out_f.create_group('data')

    out_f.create_group('data/rate')
    out_f.create_group('data/diff_rate')
    out_f.create_group('data/binned_rate')

    # numerics parameters
    for index in numerics_parameters:
        out_f.create_dataset('numerics/'+str(index), data=numerics_parameters[index])

    for index in physics_parameters:
        out_f.create_dataset('particle_physics/'+str(index), data=physics_parameters[index])

    hdf5_write_dict(out_f, 'particle_physics/dm_properties', dm_properties_dict)
    hdf5_write_dict(out_f, 'particle_physics/c_coeffs', c_dict)

    total_rate_dict = {}
    total_diff_rate_dict = {}
    total_binned_rate_dict = {}

    for i in range(len(all_total_rate_list)):
        for j in range(len(all_total_rate_list[i])):

            mass_index = int(all_total_rate_list[i][j][0][0])
            str_time_index = str(int(all_total_rate_list[i][j][0][1]))

            mass = dm_properties_dict['mass_list'][mass_index]
            
            # create dict if it doesn't exist
            if str_time_index not in total_rate_dict:
                total_rate_dict[str_time_index] = {}

            if mass_index not in total_rate_dict[str_time_index]:
                total_rate_dict[str_time_index][mass_index] = 0
            
            ###

            if str_time_index not in total_diff_rate_dict:
                total_diff_rate_dict[str_time_index] = {}
            
            if mass_index not in total_diff_rate_dict[str_time_index]:
                total_diff_rate_dict[str_time_index][mass_index] = \
                        np.zeros(len(all_diff_rate_list[i][j][1]))
            
            ###

            if str_time_index not in total_binned_rate_dict:
                total_binned_rate_dict[str_time_index] = {}

            if mass_index not in total_binned_rate_dict[str_time_index]:
                total_binned_rate_dict[str_time_index][mass_index] = \
                        np.zeros(len(all_binned_rate_list[i][j][1]))
            
            total_rate_dict[str_time_index][mass_index] += \
                    all_total_rate_list[i][j][1]
            
            total_diff_rate_dict[str_time_index][mass_index] += \
                    all_diff_rate_list[i][j][1]
            
            total_binned_rate_dict[str_time_index][mass_index] += \
                    all_binned_rate_list[i][j][1]
            
    hdf5_write_dict(out_f, 'data/rate', total_rate_dict)
    hdf5_write_dict(out_f, 'data/binned_rate', total_binned_rate_dict)
    hdf5_write_dict(out_f, 'data/diff_rate', total_diff_rate_dict)
    
    out_f.close()

def hdf5_write_output_q(out_file, numerics_parameters, physics_parameters, dm_properties_dict, c_dict,
                        all_total_rate_list, n_proc, material, all_diff_rate_list,
                        all_binned_rate_list):
    """
        
        Write data to hdf5 file

    """

    out_f = h5py.File(out_file, "w")

    out_f.create_group('numerics')
    out_f.create_group('particle_physics')
    out_f.create_group('material_properties')
    out_f.create_group('data')

    out_f.create_group('data/rate')
    out_f.create_group('data/diff_rate')
    out_f.create_group('data/binned_rate')

    # numerics parameters
    for index in numerics_parameters:
        out_f.create_dataset('numerics/'+str(index), data=numerics_parameters[index])

    for index in physics_parameters:
        out_f.create_dataset('particle_physics/'+str(index), data=physics_parameters[index])

    hdf5_write_dict(out_f, 'particle_physics/dm_properties', dm_properties_dict)
    hdf5_write_dict(out_f, 'particle_physics/c_coeffs', c_dict)

    total_rate_dict = {}
    total_diff_rate_dict = {}
    total_binned_rate_dict = {}

    for i in range(len(all_total_rate_list)):
        for j in range(len(all_total_rate_list[i])):

            mass_index = int(all_total_rate_list[i][j][0][0])
            str_time_index = '0'

            mass = dm_properties_dict['mass_list'][mass_index]
            
            # create dict if it doesn't exist
            if str_time_index not in total_rate_dict:
                total_rate_dict[str_time_index] = {}

            if mass_index not in total_rate_dict[str_time_index]:
                total_rate_dict[str_time_index][mass_index] = 0
            
            ###

            if str_time_index not in total_diff_rate_dict:
                total_diff_rate_dict[str_time_index] = {}
            
            if mass_index not in total_diff_rate_dict[str_time_index]:
                total_diff_rate_dict[str_time_index][mass_index] = \
                        np.zeros(len(all_diff_rate_list[i][j][1]))
            
            ###

            if str_time_index not in total_binned_rate_dict:
                total_binned_rate_dict[str_time_index] = {}

            if mass_index not in total_binned_rate_dict[str_time_index]:
                total_binned_rate_dict[str_time_index][mass_index] = \
                        np.zeros(len(all_binned_rate_list[i][j][1]))
            
            total_rate_dict[str_time_index][mass_index] += \
                    all_total_rate_list[i][j][1]
            
            total_diff_rate_dict[str_time_index][mass_index] += \
                    all_diff_rate_list[i][j][1]
            
            total_binned_rate_dict[str_time_index][mass_index] += \
                    all_binned_rate_list[i][j][1]
            
    hdf5_write_dict(out_f, 'data/rate', total_rate_dict)
    hdf5_write_dict(out_f, 'data/binned_rate', total_binned_rate_dict)
    hdf5_write_dict(out_f, 'data/diff_rate', total_diff_rate_dict)
    
    out_f.close()
