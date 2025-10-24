# Matrix Multiplication with CUDA
Starter code for CSE260 HW2.

You will only need to edit files in src_todo folder.
Run all your commands from the root directory of the git repo.

## Commands to build and run

To build the binary:
```bash
make `cat src_todo_T4/OPTIONS.txt`
```

To build the binary with cublas:
```bash
make cublastest=1 `cat src_todo_T4/OPTIONS.txt`
```

To clean the output files of the make command:
```bash
make clean
```

> [!IMPORTANT]
> On the AWS g4dn.xlarge instance run the following command before running any experiments. It sets fixed GPU clock rates to ensure consistent performance
> ```bash
> sudo nvidia-smi -pm 1
> sudo nvidia-smi -ac 5001,1590
> ```
To Run:
```bash
./mmpy `cat src_todo_T4/OPTIONS_RUNTIME.txt` -n 256
```

## Additional Useful Commands
To run Script in tools folder:
```bash
./tools/run_ncu.sh
```

If you get Permission denied error when executing a file:
```bash
chmod +x name_of_file
```
eg: `chmod +x tools/*`

Find GPU chipset:
```bash
lspci | grep -i --color 'vga\|3d\|2d'
```