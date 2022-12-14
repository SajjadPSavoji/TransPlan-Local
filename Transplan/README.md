# Transplan
my implementations for the transplan project

## Install
IMPORTANT: do not install opencv via pip as it will have conflict with Qt5 dependencies. Install it using apt/apt-get
```
apt install python3-opencv
apt install curl
```
grab the appropriate conda version from [here](https://www.anaconda.com/products/distribution)
copy the link for installer and run
```code
curl -O <link_to_installer>
bash Anaconda*.sh
```
reopen your terminal for changes to be effective.
for updating the conda run
```code
conda update conda
conda update anaconda
conda config --set auto_activate_base false
```
