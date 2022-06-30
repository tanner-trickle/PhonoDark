io_parameters = {
        'output_folder'       : 'data',
        'output_filename_extra' : '50x10x10_noborn'
}

numerics_parameters = {
        # number of grid points in radial direction
        'n_a'             : 50,
        # number of grid points in theta direction
        'n_b'             : 10,
        # number of grid points in phi direction
        'n_c'             : 10,
        # sampling parameter in radial direction
        'power_a'         : 2,
        # sampling parameter in theta direction
        'power_b'         : 1,
        # sampling parameter in phi direction
        'power_c'         : 1,
        # number of grid points in x direction when computing the Debye-Waller factor
        'n_DW_x'          : 10,
        # number of grid points in y direction when computing the Debye-Waller factor
        'n_DW_y'          : 10,
        # number of grid points in z direction when computing the Debye-Waller factor
        'n_DW_z'          : 10,
        # width of the energy bins
        'energy_bin_width': 10**(-3),
        # number of energy bins
        'n_E_bins': 100,
        # option to automatically cut off the q integral when q is much greater
        # than the Debye Waller factor
        'q_cut'           : True,
        # option to use a predefined special mesh which is optimal when the scattering
        # potential is a power in q
        'special_mesh'    : True
}
