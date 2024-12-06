#!/bin/bash
cd ..
MM_path=./MM
input="./data/matrix_list21.csv"
{
  read
  i=1
  while IFS=',' read -r mid Group Name rows cols nonzeros
  do
    echo "$mid $Group $Name $rows $cols $nonzeros"
    ./spmv_double  $MM_path/$Group/$Name/$Name.mtx 0
    ./spmv_half  $MM_path/$Group/$Name/$Name.mtx 0
    i=`expr $i + 1`
  done 
} < "$input"