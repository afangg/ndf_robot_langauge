# uncomment and run for installing torch geometric stuff
export LD_LIBRARY_PATH="~/miniconda3/envs/polymetis-ndf-llm/lib:$LD_LIBRARY_PATH"
pip install --no-index --no-cache-dir -v torch_scatter -f https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html
pip install --no-index --no-cache-dir -v torch_sparse -f https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html
pip install --no-index --no-cache-dir -v torch_cluster -f https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html
pip install --no-index --no-cache-dir -v torch_spline_conv -f https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html
pip install torch-geometric==2.0.2
