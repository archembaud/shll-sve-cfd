#!/bin/sh
# Use this to install software packages
yum groupinstall 'Development Tools' -y
amazon-linux-extras enable python3.8
yum install -y python38 python38-devel
# Now try installing the version of pip compatible with 3.8
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# Install
python3.8 get-pip.py
# Install tmux
yum install tmux -y
pip3 install boto3
