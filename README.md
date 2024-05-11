# GANN
Graph Attention Neural Networks for Mapping Materials and Molecules


Our training is carried out on a system with 10 CPU cores, 32 GB of RAM, and an NVIDIA GeForce GTX 2080 Ti GPU.
Check requirements.txt for other dependencies.


We provide an implementation for the QMOF experiment. 


1. Data

   
(a) Download database file from https://github.com/arosen93/QMOF or https://figshare.com/articles/dataset/QMOF_Database/13147324
(b) Find 'opt-geometries.xyz' from the downloaded file and place it in /GANN/data/
(c) cd /GANN/data/
(c) Run the python script to process into structure files with "python xyz_to_cifs.py"

3. Training 
To train the model in the paper, run these commands:
cd /GANN
python main.py --batch-size 32 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1  --workers 1 --epochs 200 --print-freq 1 data > log.out

4. Evaluation
All results are available in the file 'output'.
