#!/bin/bash
bash run_spmv_all.sh
python3 process_data.py
python3 f16_bar.py
python3 f64_bar.py
python3 f16_scatter.py
python3 f64_scatter.py
python3 preprocess_scatter.py
