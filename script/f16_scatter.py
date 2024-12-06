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
# 7:cusparse_pre_time
# 8:cusparse_spmv_time
# 9:cusparse_spmv_gflops

filename = '../data/a100_f16_record.csv'
fn_split = re.split('[/_.]+', filename)
figname = fn_split[2] + '_' + fn_split[3] + '_scatter.pdf'

data_from_csv=pd.read_csv(filename, usecols=[3,6], names=['compute_nnz','perf_dasp'])
cusp_from_csv=pd.read_csv(filename, usecols=[3,9], names=['compute_nnz','perf_cusp'])

font = {'weight' : 'normal',
'size'   : 36 ,}
fig, axs = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
plt.subplot(2,1,1)

plt.scatter(np.log10(cusp_from_csv.compute_nnz), cusp_from_csv.perf_cusp,s=50,c='#4ea59f',marker='o',linewidth='0.0',label='cuSPARSE (v12.0)')
plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.perf_dasp,s=50,c='#ee6a5b',marker='o',linewidth='0.0',label='DASP (this work)')

plt.legend(loc="upper left",fontsize=32,markerscale=1.5)

plt.ylabel("Performance (Gflops)",font, labelpad=20)
plt.ylim(0,600)
plt.xlim(0,9)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(labelsize=24)

plt.subplot(2,1,2)

plt.plot([0, 10], [1, 1], color = 'black', linewidth=2, linestyle='-')
plt.scatter(np.log10(data_from_csv.compute_nnz), data_from_csv.perf_dasp/cusp_from_csv.perf_cusp,s=50,c='#f6b654',marker='o',linewidth='0.0',label='This work vs. cuSPARSE')


plt.ylim(0,5)
plt.xlim(0,9)
plt.ylabel("Speedup\nDASP over cuSPARSE",fontsize=32,labelpad=20)
plt.xlabel("#nonzeros of matrix (log10 scale)",fontsize=32,labelpad=20)
plt.grid(c='grey',alpha=0.8,linestyle='--')
plt.tick_params(labelsize=24)

gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1]) 
plt.rcParams["figure.figsize"] = (10.5,6.5)

plt.tight_layout()
plt.subplots_adjust(left = 0.14, right = 0.99, wspace = 0.1, hspace= 0.1)

plt.savefig(figname, dpi=200)

# plt.show()
