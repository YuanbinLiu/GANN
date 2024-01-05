Our training is carried out on a system with 10 CPU cores, 32 GB of RAM, and an NVIDIA GeForce GTX 2080 Ti GPU.
Check requirements.txt for other dependencies. The installation takes about ten minutes.  

We provide an implementation for the QMOF experiment. 
==================
1. Data 
(a) Download database file from https://github.com/arosen93/QMOF or https://figshare.com/articles/dataset/QMOF_Database/13147324
(b) Find 'opt-geometries.xyz' from the downloaded file and just place it in /MGANN/data/
(c) cd /MGANN/data/
(d) Run the python script to process into structure files with "python xyz_to_cifs.py"

==================
2. Training 
To train the model in the paper (for reproduction), run these commands:
cd /MGANN
python main.py --batch-size 32 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1  --workers 1 --epochs 200 --print-freq 1 data > log.out

==================
3. Evaluation
All results are available in the file 'output'.


++++++++++++++++++
Demo
We provide a toy example to test if the package is installed and working. The data have been ready in /MGANN/data/demo for training (including 100 datasets). Procedure below:
(a)Go to /MGANN and type 
python main.py --batch-size 10 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1  --workers 1 --epochs 20 --print-freq 1 data > log.out
(b)One can expect the training to end in 10 seconds on a system with 10 CPU cores, 32 GB of RAM, and an NVIDIA GeForce GTX 2080 Ti GPU.