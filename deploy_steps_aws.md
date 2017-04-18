## Deploy Steps for AWS Ubuntu 14.04 LTS EC2 Instance

### Login to AWS Instance:

`ssh -i <your AWS Pem key file> ubuntu@<aws ip>`


### Install Python / Git

```
sudo apt-get update
sudo apt-get upgrade

# Install GIT
sudo apt-get install git

# Install Anaconda (Miniconda)
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

# To update the path (Make sure you said Yes when it asked to update path in the Miniconda install steps)
source ~/.bashrc
```

### Download App Source Code:

```
git clone https://github.com/sampathweb/apparel-styles.git


cd apparel-styles
python env/create_env.py
source activate env/venv
python env/install_packages.py

python ml_src/build_model.py
python run.py (Confirm that App is running)


sudo apt-get install supervisor
sudo vi /etc/supervisor/conf.d/apparel-styles.conf
<press i insert mode>

[program:apparel-styles]
autorestart = true
command = /home/ubuntu/apparel-styles/env/venv/bin/python /home/ubuntu/apparel-styles/run_server.py --debug=False --port=80
numprocs = 1
startsecs = 10
stderr_logfile = /var/log/supervisor/apparel-styles.log
stdout_logfile = /var/log/supervisor/apparel-styles.log
environment = PYTHONPATH="/home/ubuntu/apparel-styles/env/bin/"

<escape :wq>

sudo supervisorctl reload

```

### Test the App

Open Browser:  `http://<AWS IP>` (App is Live!)
