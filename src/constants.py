"""
    Physical constants used throughout the rest of the program.

    Natural (eV) units are used unless otherwise specified.
"""

import numpy as np
from scipy import special

# electron mass
M_ELEC                   = 511*10**3

# proton mass
M_NUCL                   = 0.938*10**9

# Conversion factors ###

THz_To_eV                = 6.58*10**(-4)
PI                       = np.pi
Ang_To_inveV             = 5.06773*10**(-4)
fm_To_inveV              = 5.06773*10**(-9)
AMU_To_eV                = 9.31*10**8
eV_to_AMU                = 1.074*10**-9
g_over_cm3_to_eV_over_A3 = 5.6095*10**8
kg_to_eV                 = 5.6095*10**35
second_to_one_over_eV    = 1.51976*10**15
inveV_to_cm              = 1.973*10**-5
invAng_to_eV             = 1.97*10**3
invcmet_To_eV            = 1.973*10**(-5)
GeV_To_eV                = 10**9
inveV_To_invGeV          = 10**9
kmet_per_sec_to_none     = 3.34*10**(-6)

#####

# fine structure constant
ALPHA_EM                 = 1.0/137.0

# dark matter density
RHO_DM                   = 0.4*GeV_To_eV*invcmet_To_eV**3

# 1 kg x year exposure
KG_YR                    = 2.69*10**58

# angle (radians) of the North pole relative to the Earth velocity
THETA_E                  = 42*(PI/180)

# Maxwell Boltzmann velocity distribution parameters
V0   = 230*kmet_per_sec_to_none
VE   = 240*kmet_per_sec_to_none
VESC = 600*kmet_per_sec_to_none

N0 = PI**(3/2)*V0**2*(V0*special.erf(VESC/V0) - (2/np.sqrt(PI))*VESC*np.exp(-VESC**2/V0**2))
C1 = PI*V0**2/N0
C2 = np.exp(-(VESC/V0)**2)

# keep track of the supercells used when generating phonon files
# supercell_data = {
# 	'ZnS'     : [2, 2, 2],
# 	'CsI'     : [2, 2, 2],
# 	'GaAs'    : [2, 2, 2],
# 	'SiO2'    : [3, 3, 3],
# 	'Al2O3'   : [2, 2, 2],
# 	'InSb'    : [2, 2, 2],
# 	'LiF'     : [2, 2, 2],
# 	'NaCl'    : [2, 2, 2],
# 	'MgO'     : [2, 2, 2],
# 	'GaSb'    : [2, 2, 2],
# 	'NaI'     : [2, 2, 2],
# 	'PbS'     : [2, 2, 2],
# 	'PbSe'    : [2, 2, 2],
# 	'PbTe'    : [2, 2, 2],
# 	'CaF2'    : [3, 3, 3],
# 	'AlN'     : [3, 3, 2],
# 	'CaWO4'   : [2, 2, 1],
# 	'MgF2'    : [2, 2, 2],
# 	'ZnO'     : [2, 2, 2],
# 	'NaF'     : [2, 2, 2],
# 	'GaN'     : [3, 3, 2],
# 	'Al2O3_db': [2, 2, 1],	
# 	'Si'      : [2, 2, 2],
# 	'hBN'     : [2, 2, 2],	
#         'YGG'     : [1, 1, 1]
# }
