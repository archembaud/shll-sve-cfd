#!/bin/sh
# Prepare yum update
sudo yum update
# Use this to install software packages
yum groupinstall 'Development Tools' -y
sudo yum -y install environment-modules python3 glibc-devel tmux
sudo yum -y install 'dnf-command(config-manager)' procps psmisc make environment-modules
sudo yum config-manager --add-repo https://developer.arm.com/packages/ACfL:AmazonLinux-2023/latest/ACfL:AmazonLinux-2023.repo
sudo yum -y install acfl
# Now try installing the version of pip compatible with installed python
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# Install
python3 get-pip.py
pip3 install boto3
# Set modules up for all users
echo ". /usr/share/Modules/init/bash" >> /etc/bashrc
echo "module use /opt/arm/modulefiles" >> /etc/bashrc
