wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda create -n py312 python=3.12
conda activate py312
conda install python=3.12
conda install conda-forge::xvfbwrapper
conda install conda-forge::ffmpeg
python3.12 -m pip install --user pipx
python3.12 -m pipx ensurepath
python3.12 -m pipx install uv

mkdir -p ~/workspace
cd ~/workspace
git clone --branch 
cd Bachelor-Thesis

uv sync

deactivate
