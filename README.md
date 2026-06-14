# Computational Imaging — Group A

## Project: Sparse-Views CT Reconstruction

### Methods
- Total Variation (TV) regularization
- End-to-end neural network (UNet)
- Plug-and-Play with Half Quadratic Splitting (HQS)

### How to run
0. Create conda environment: `conda create -n ct_gpu python=3.10 anaconda && conda activate ct_gpu`
1. Install dependencies: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`
2. Preprocess data: While in sparse_mayo_light/ :
`python3 scripts/preprocess.py`
3. Generate sinograms: 
`python3 scripts/generate_sinograms.py`
4. Run lambda choice and nr of epochs choice:
`python3 scripts/lambda_choice.py`
`python3 scripts/tv_loss_vs_epochs.py`
5. Run a method: `python3 scripts/run_tv.py` , `python3 scripts/run_end_to_end_2.py` , `python3 scripts/run_pnp.py`
