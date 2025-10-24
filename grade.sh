#!/bin/bash

N_values=(256 512 1024 1025 2047 2048)

for N in "${N_values[@]}"; do
    output=$(./mmpy $(cat src_todo_T4/OPTIONS_RUNTIME.txt) -n $N)

    passed=$(echo "$output" | grep -q "answers matched to within" && echo "yes" || echo "no")
    if [ "$passed" != "yes" ]; then
        echo "Test failed for N=$N"
        continue
    fi

    gflops=$(echo "$output" | grep "Device computation time" | sed 's/.*\[//; s/ gflops\]//')

    echo "N: $N, GFLOPS: $gflops"
done