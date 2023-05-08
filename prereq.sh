
apt-get update && apt-get install -y libgl1-mesa-glx
sudo apt-get install python3.6
apt-get install -y libglib2.0-0

# /opt/render/project/src/.venv/bin/python -m pip install --upgrade pip

python -m pip install --upgrade pip

pip install -r requirements.txt



fallocate -l 512M /swapfile
chmod 600 /swapfile

dd if=/dev/zero of=/swapfile bs=1024 count=524288
chown root:root /swapfile
chmod 0600 /swapfile
mkswap /swapfile
swapon /swapfile
pip --no-cache-dir  install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


pip --no-cache-dir install ultralytics


export CUDA_VISIBLE_DEVICES=''
export TF_ENABLE_TENSORRT=0
export TF_DISABLE_DEPRECATION_WARNINGS=1
export TF_ENABLE_ONEDNN_OPTS=0


pip install tensorflow-cpu

pip uninstall -y opencv-contrib-python
# pip uninstall -y opencv-python

pip install --force-reinstall opencv-python-headless==4.5.4.58


find ~ -name "matplotlib"

rm -rf ~/.matplotlib
rm -rf ~/.cache/matplotlib
rm -rf ~/.cache/fontconfig/
