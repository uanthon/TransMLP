# TransMLP
# Install libs step by step
conda create -n Transmlp python=3.7 -y\
conda activate Transmlp\
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y\
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2\
pip install pointnet2_ops_lib/.\\

# Make data folder and download the dataset
cd leaf_steam\
mkdir data\
cd data\
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate\
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip\

# Run
python main.py --model TransMLP
