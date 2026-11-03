wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda create -n py313 python=3.13
conda activate py313
conda install python=3.13
conda install conda-forge::xvfbwrapper
conda install conda-forge::ffmpeg
python3.13 -m pip install --user pipx
python3.13 -m pipx ensurepath
python3.13 -m pipx install uv

mkdir -p ~/workspace
cd ~/workspace
git clone --branch 
cd Bachelor-Thesis

uv sync

deactivate
