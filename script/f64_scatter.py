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
fn_split = re.split('[/_.]+', filename)
figname = fn_split[2] + '_' + fn_split[3] + '_scatter.pdf'

data_from_csv=pd.read_csv(filename, usecols=[3,6,9,12,15,16,17,18], names=['compute_nnz','perf_dasp','perf_cusp','perf_csr5','perf_tile','speedup1','speedup2','speedup3'])

font = {'weight' : 'normal',
'size'   : 36 ,}

# fig=plt.figure(figsize=(16, 12))
fig, axs = plt.subplots(4, 1, figsize=(16, 22), gridspec_kw={'height_ratios': [2,0.7,0.7,0.7]})

plt.subplot(4,1,1)

plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.perf_csr5,s=50,c='#f6b654',marker='o',linewidth='0.0',label='CSR5')
plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.perf_tile,s=50,c='#c7dbd5',marker='o',linewidth='0.0',label='TileSpMV')
plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.perf_cusp,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='cuSPARSE v12.0')
plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.perf_dasp,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='DASP (this work)')

plt.legend(loc="upper left",fontsize=32,markerscale=1.5)

plt.ylabel("Performance (Gflops)",font, labelpad=20)
plt.ylim(0,270)
plt.xlim(0,9)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(labelsize=24)

plt.subplot(4,1,2)
plt.plot([0, 9], [1, 1], color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.speedup2,s=50,c='#8c9976',marker='o',linewidth='0.0')
plt.ylim(0,5)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over\nCSR5",fontsize=32,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=24)
plt.tick_params(axis='x',labelsize=24,labelcolor='w')

plt.subplot(4,1,3)
plt.plot([0, 9], [1, 1], color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.speedup3,s=50,c='#8c9976',marker='o',linewidth='0.0')
plt.ylim(0,5)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over\nTileSpMV",fontsize=32,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(axis='y',labelsize=24)
plt.tick_params(axis='x',labelsize=24,labelcolor='w')

plt.subplot(4,1,4)
plt.plot([0, 9], [1, 1], color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.speedup1,s=50,c='#8c9976',marker='o',linewidth='0.0')
plt.ylim(0,4)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over\ncuSPARSE",fontsize=32,labelpad=20)
plt.xlabel("#nonzeros of matrix (log10 scale)",fontsize=32,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(labelsize=24)



plt.tight_layout()
plt.subplots_adjust(left = 0.14, right = 0.99, wspace = 0.1, hspace= 0.1)

plt.savefig(figname, dpi=200)

# plt.show()
