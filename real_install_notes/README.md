# Main steps
Main conda install
```
conda env create -n polymetis-ndf-llm-dep python=3.8 --file real_env_conda.yml
conda install cudatoolkit=11.3
```

Then, pip install
```
pip install -r real_env_pip.txt
```

Then, install the repo
```
cd ..
pip install -e .
```

# Other steps, if needed
If `torch-geometric` packages needed, modify `real_env_torch.txt` with path to conda env, and uncomment lines + run to install 
If `lcm` and/or `detectron2` needed (instance segmentation), download source code and follow paths to `lcm/lcm-python` (run `pip install .`) or detectron2 (run `pip install -e .`)

