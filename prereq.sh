
pip3 install -r requirements.txt

dd if=/dev/zero of=/swapfile bs=1024 count=524288
chown root:root /swapfile
chmod 0600 /swapfile
mkswap /swapfile
swapon /swapfile
pip3 --no-cache-dir  install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


pip3 --no-cache-dir install ultralytics
pip3 install tensorflow-cpu
pip3 install asyncio

pip3 uninstall opencv-contrib-python
pip3 uninstall opencv-python

pip3 install --force-reinstall opencv-python-headless==4.5.4.58
