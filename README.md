# Computational Imaging — Group A

## Project: Sparse-Views CT Reconstruction

### Methods
- Total Variation (TV) regularization
- End-to-end neural network (UNet)
- Plug-and-Play with Half Quadratic Splitting (HQS)

### How to run
0. Create conda environment: `conda create -n ct_gpu python=3.10 anaconda && conda activate ct_gpu`
1. Install dependencies: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`
2. Preprocess data: 
`cd scripts && python preprocess.py`
3. Run a method: `python scripts/run_tv.py`
4. Compare all results: `python scripts/compare_all.py`
