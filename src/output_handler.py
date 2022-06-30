import h5py
import numpy as np
import os

def hdf5_write_dict(hdf5_file, group_name, data_dict):
    """
        Recursively go through a dictionary and write all values to the group_name
    """

    for index in data_dict:

        if type(data_dict[index]) is dict:

            hdf5_write_dict(hdf5_file, group_name+'/'+str(index), data_dict[index])

        else:

            # this input dataset is not an array, therefore there is no default way to save it
            if index not in ['phonopy_config', 'c_dict_form']:

                hdf5_file.create_dataset(group_name+'/'+str(index), data=data_dict[index])

def create_output_filename(inputs):
    """
        Creates the output filename.
    """

    out_filename = os.path.join(
            inputs['numerics']['io_parameters']['output_folder'], 
            inputs['material']['name']+
                    '_'+inputs['mod_names']['p']+
                    '_'+inputs['mod_names']['n']+
                    '_'+inputs['numerics']['io_parameters']['output_filename_extra']+
                    '.hdf5')

    return out_filename

def save_output(filename, inputs, binned_rate, version_number):
    """
        
        binned_rate - [n_masses, n_vE, n_E_bins]

    """

    out_f = h5py.File(filename, "w")

    # version number
    out_f.create_dataset('version', data=version_number)

    # inputs
    hdf5_write_dict(out_f, 'inputs', inputs)

    # computed data
    out_f.create_group('data')

    out_f.create_group('data/binned_rate')
    out_f.create_group('data/total_rate')

    for m in range(len(binned_rate)):
        for v in range(len(binned_rate[0])):

            b_rate = binned_rate[m, v, :]
            total_rate = np.sum(b_rate)

            data_path = 'data/v_'+f'{v}/m_'+f'{m}'

            if data_path not in out_f:
                out_f.create_group(data_path)

            out_f.create_dataset(data_path+'/binned_rate', data=b_rate)
            out_f.create_dataset(data_path+'/total_rate', data=total_rate)

    out_f.close()
