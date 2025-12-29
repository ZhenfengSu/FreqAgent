python --version &&
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" &&
nvcc -V