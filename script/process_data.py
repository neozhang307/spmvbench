import numpy as np
from scipy.stats.mstats import gmean

with open("../data/a100_f64_record.csv", 'r') as f:
    my_performance = {}
    cusparse_performance = {}
    for line in f.readlines():
        # print(line)
        name, m, n, nnz, pretime, spmv_time, gflops, cu_pretime, cu_spmv_time, cu_gflops = line.strip().split(',')
        my_performance[name] = {
            'm': m,
            'n': n,
            'nnz': nnz,
            'pretime': pretime,
            'spmv_time': spmv_time,
            'gflops': gflops
        }

        cusparse_performance[name] = {
            'm': m,
            'n': n,
            'nnz': nnz,
            'pretime': cu_pretime,
            'spmv_time': cu_spmv_time,
            'gflops': cu_gflops
        }

with open("../csr5_record.csv", 'r') as f:
    csr5_performance = {}
    for line in f.readlines():
        name, m, n, nnz, pretime, spmv_time, gflops = line.strip().split(',')        
        csr5_performance[name] = {
            'm': m,
            'n': n,
            'nnz': nnz,
            'pretime': pretime,
            'spmv_time': spmv_time,
            'gflops': gflops
        }

with open("../tilespmv_record.csv", 'r') as f:
    tilespmv_performance = {}
    for line in f.readlines():
        name, m, n, nnz, pretime, spmv_time, gflops = line.strip().split(',')        
        tilespmv_performance[name] = {
            'm': m,
            'n': n,
            'nnz': nnz,
            'pretime': pretime,
            'spmv_time': spmv_time,
            'gflops': gflops
        }

total_cusp = []
total_csr5 = []
total_tile = []
total_half_cusp = []
valid_cusp = 0
valid_csr5 = 0
valid_tile = 0
valid_half_cusp = 0
num_cusp = 0
num_csr5 = 0
num_tile = 0
num_half_cusp = 0

with open('../data/a100_f64_all.csv', 'w') as f:
    for name in my_performance:
        dasp_gflop = (float)(list(my_performance[name].values())[5])
        valid_cusp += 1
        if name in csr5_performance and name in tilespmv_performance:
            my_value = list(my_performance[name].values())
            cusparse_value = list(cusparse_performance[name].values())[-3:]
            csr5_value = list(csr5_performance[name].values())[-3:]
            tile_value = list(tilespmv_performance[name].values())[-3:] 
            cusp_gflop = (float)(list(cusparse_performance[name].values())[5])
            csr5_gflop = (float)(list(csr5_performance[name].values())[5])
            tile_gflop = (float)(list(tilespmv_performance[name].values())[5])
            speedup1 = dasp_gflop / cusp_gflop 
            total_cusp.append(speedup1)
            speedup2 = dasp_gflop / csr5_gflop
            total_csr5.append(speedup2)
            speedup3 = dasp_gflop / tile_gflop
            total_tile.append(speedup3)
            valid_csr5 += 1
            valid_tile += 1
            if speedup1 > 1:
                num_cusp += 1
            if speedup2 > 1:
                num_csr5 += 1
            if speedup3 > 1:
                num_tile += 1
            print(f'{name},{",".join(my_value)},{",".join(cusparse_value)},{",".join(csr5_value)},{",".join(tile_value)},{str(speedup1)+","+str(speedup2)+","+str(speedup3)}', file=f)
        if name in csr5_performance and name not in tilespmv_performance:
            my_value = list(my_performance[name].values())
            cusparse_value = list(cusparse_performance[name].values())[-3:]
            csr5_value = list(csr5_performance[name].values())[-3:]
            cusp_gflop = (float)(list(cusparse_performance[name].values())[5])
            csr5_gflop = (float)(list(csr5_performance[name].values())[5])
            speedup1 = dasp_gflop / cusp_gflop 
            total_cusp.append(speedup1)
            speedup2 = dasp_gflop / csr5_gflop
            total_csr5.append(speedup2)
            speedup3 = 100
            valid_csr5 += 1
            if speedup1 > 1:
                num_cusp += 1
            if speedup2 > 1:
                num_csr5 += 1
            print(f'{name},{",".join(my_value)},{",".join(cusparse_value)},{",".join(csr5_value)},{",,"},{str(speedup1)+","+str(speedup2)+","+str(speedup3)}', file=f)
        if name not in csr5_performance and name in tilespmv_performance:
            my_value = list(my_performance[name].values())
            cusparse_value = list(cusparse_performance[name].values())[-3:]
            tile_value = list(tilespmv_performance[name].values())[-3:]
            cusp_gflop = (float)(list(cusparse_performance[name].values())[5])
            tile_gflop = (float)(list(tilespmv_performance[name].values())[5])
            speedup1 = dasp_gflop / cusp_gflop 
            total_cusp.append(speedup1)
            speedup2 = 100
            speedup3 = dasp_gflop / tile_gflop
            total_tile.append(speedup3)
            valid_tile += 1
            if speedup1 > 1:
                num_cusp += 1
            if speedup3 > 1:
                num_tile += 1
            print(f'{name},{",".join(my_value)},{",".join(cusparse_value)},{",,"},{",".join(tile_value)},{str(speedup1)+","+str(speedup2)+","+str(speedup3)}', file=f)
        if name not in csr5_performance and name not in tilespmv_performance:
            my_value = list(my_performance[name].values())
            cusparse_value = list(cusparse_performance[name].values())[-3:]
            cusp_gflop = (float)(list(cusparse_performance[name].values())[5])
            speedup1 = dasp_gflop / cusp_gflop 
            total_cusp.append(speedup1)
            speedup2 = 100
            speedup3 = 100
            if speedup1 > 1:
                num_cusp += 1
            print(f'{name},{",".join(my_value)},{",".join(cusparse_value)},{",,,,,"},{str(speedup1)+","+str(speedup2)+","+str(speedup3)}', file=f)


with open("../data/a100_f16_record.csv", 'r') as f:
    half_dasp_performance = {}
    half_cusp_performance = {}
    for line in f.readlines():
        # print(line)
        valid_half_cusp += 1
        name, m, n, nnz, pretime, spmv_time, gflops, cu_pretime, cu_spmv_time, cu_gflops = line.strip().split(',')
        half_dasp_performance[name] = {
            'm': m,
            'n': n,
            'nnz': nnz,
            'pretime': pretime,
            'spmv_time': spmv_time,
            'gflops': gflops
        }

        half_cusp_performance[name] = {
            'm': m,
            'n': n,
            'nnz': nnz,
            'pretime': cu_pretime,
            'spmv_time': cu_spmv_time,
            'gflops': cu_gflops
        }

        speedup = (float)(list(half_dasp_performance[name].values())[5]) / (float)(list(half_cusp_performance[name].values())[5])
        total_half_cusp.append(speedup)
        if speedup > 1:
            num_half_cusp += 1
    

with open("../result.txt", 'w') as f:
    print(f'{"Experimental results"}', file=f)
    print(f'{"For double precision:"}', file=f)
    print(f'{"DASP has "+str(num_cusp)+" (valid num: "+str(valid_cusp)+") matrices faster than cuSPARSE, the geomean is "+str(gmean(total_cusp))}', file=f)
    print(f'{"DASP has "+str(num_csr5)+" (valid num: "+str(valid_csr5)+") matrices faster than CSR5, the geomean is "+str(gmean(total_csr5))}', file=f)
    print(f'{"DASP has "+str(num_tile)+" (valid num: "+str(valid_tile)+") matrices faster than TileSpMV, the geomean is "+str(gmean(total_tile))}', file=f)
    print(f'{"For half precision:"}', file=f)
    print(f'{"DASP has "+str(num_half_cusp)+" (valid num: "+str(valid_half_cusp)+") matrices faster than cuSPARSE, the geomean is "+str(gmean(total_half_cusp))}', file=f)