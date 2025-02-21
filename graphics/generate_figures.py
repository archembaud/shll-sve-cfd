'''
Generate figures using matplotlib for use
in presentations and publications.
'''
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm

one_dimension_cell_counts = ['256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '65536']

# Timings for base C codes without optimization (GCC) - Table 1
base_c_no_opt_1D = np.array([ [0.006, 0.006], [0.021, 0.021], [0.082, 0.082], [0.332, 0.330], [1.349, 1.349], [5.404, 5.286], [21.584, 21.669], [92.176, 91.594], [367.05, 367.117]])
base_c_no_opt_1D_avg = np.average(base_c_no_opt_1D, axis=1)
# Timings for base C codes with max optimization (GCC) - Table 2
base_c_max_opt_1D = np.array([ [0.002, 0.002], [0.007, 0.007], [0.025, 0.025], [0.099, 0.099], [0.404, 0.403], [1.597, 1.595], [6.320, 6.312], [29.466, 29.495], [121.902, 121.748]  ])
base_c_max_opt_1D_avg = np.average(base_c_max_opt_1D, axis=1)
# Timings for SVE vector code (GCC) - Table 3
sve_c_no_opt_1D = np.array([  [0.004, 0.004], [0.016, 0.016], [0.06, 0.06], [0.235, 0.234], [0.941, 0.943], [3.756, 3.752], [15.018, 15.003], [63.542, 63.792], [257.973, 257.974] ])
sve_c_no_opt_1D_avg = np.average(sve_c_no_opt_1D, axis=1)
# Max optimization - Table 4
sve_c_max_opt_1D = np.array([ [	0.001, 0.001], [0.004, 0.004], [0.014, 0.014], [0.057, 0.057], [0.222, 0.231], [0.892, 0.892], [3.488, 3.478], [17.834, 17.818], [71.269, 71.268] ])
sve_c_max_opt_1D_avg = np.average(sve_c_max_opt_1D, axis=1)
# Timings for base C codes (1D) without optimization using ARM Compiler - Table 5
base_c_max_opt_arm_1D = np.array([ [0.002, 0.002], [0.005, 0.005], [0.015, 0.015], [0.059, 0.059], [0.240, 0.239], [0.945, 0.945], [3.505, 3.505], [15.926, 15.933], [68.001, 67.858] ])
base_c_max_opt_arm_1D_avg = np.average(base_c_max_opt_arm_1D, axis=1)

# Make this large
matplotlib.rcParams.update({'font.size': 22})

# Generate bar graph of speedups
'''
# 1D, no optimization, gcc
base_sve_speedup_no_opt = base_c_no_opt_1D_avg/sve_c_no_opt_1D_avg
plt.bar(one_dimension_cell_counts, base_sve_speedup_no_opt)
plt.title('Speedup of sve vector instrinsics solver vs benchmark solver (1D, no optimization)')
plt.xlabel('Number of Cells')
plt.ylabel('Speedup')
plt.show()
# 1D, max optimization, gcc
base_sve_speedup_max_opt = base_c_max_opt_1D_avg/sve_c_max_opt_1D_avg
plt.bar(one_dimension_cell_counts, base_sve_speedup_max_opt)
plt.title('Speedup of sve vector instrinsics solver vs benchmark solver (1D, maximum optimization)')
plt.xlabel('Number of Cells')
plt.ylabel('Speedup')
plt.show()
'''
# 1D, max optimization, arm
base_sve_speedup_max_opt_arm = base_c_max_opt_arm_1D_avg/sve_c_max_opt_1D_avg
plt.bar(one_dimension_cell_counts, base_sve_speedup_max_opt_arm)
plt.title('Speedup of sve vector instrinsics solver vs benchmark solver (1D, maximum optimization, ARM Compiler)')
plt.xlabel('Number of Cells')
plt.ylabel('Speedup')
plt.show()