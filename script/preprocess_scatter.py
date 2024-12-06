import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

# 0 matrix 
# 1 rowA
# 2 colA
# 3 nnzA
# 4 dasp_pre_time
# 5 dasp_spmv_time
# 6 dasp_spmv_gflops
# 7:cusp_pre_time
# 8:cusp_spmv_time
# 9:cusp_spmv_gflops  
# 10:csr5_pre_time
# 11:csr5_spmv_time
# 12:csr5_spmv_gflops  
# 13:tile_pre_time
# 14:tile_spmv_time
# 15:tile_spmv_gflops  

filename = '../data/a100_f64_all.csv'

data_from_csv=pd.read_csv(filename, usecols=[3,4,7,10,13], names=['compute_nnz','dasp_pre_time','cusp_pre_time','csr5_pre_time','tile_pre_time'])

font = {'weight' : 'normal',
'size'   : 36 ,}

fig=plt.figure(figsize=(16, 12))

plt.scatter(np.log10(data_from_csv.compute_nnz), np.log10(data_from_csv.csr5_pre_time),s=50,c='#f6b654',marker='o',linewidth='0.0',label='CSR5')
plt.scatter(np.log10(data_from_csv.compute_nnz), np.log10(data_from_csv.cusp_pre_time),s=50,c='#c7dbd5',marker='o',linewidth='0.0',label='cuSPARSE v12.0')
plt.scatter(np.log10(data_from_csv.compute_nnz), np.log10(data_from_csv.tile_pre_time),s=50,c='#4ea59f',marker='o',linewidth='0.0',label='TileSpMV')
plt.scatter(np.log10(data_from_csv.compute_nnz), np.log10(data_from_csv.dasp_pre_time),s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='DASP (this work)')

# plt.legend(loc="upper left",fontsize=32,markerscale=1.5)

plt.legend(loc = 'upper left', labelspacing = 0.1, columnspacing = 0.3, fontsize=28, markerscale=1.5, ncol = 2)
plt.ylabel("Preprocessing time (ms, log10 scale)",font, labelpad=20)
plt.ylim(-3,5)
plt.xlim(0,9)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(labelsize=24)

plt.tight_layout()
plt.subplots_adjust(left = 0.14, right = 0.99, wspace = 0.1, hspace= 0.1)

plt.savefig('preprocessing_time.pdf', dpi=200)

# plt.show()
