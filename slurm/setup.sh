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
git clone https://github.com/Wizzardexe/UQ-T2I.git
cd UQ-T2I
git checkout main

uv sync

deactivate
