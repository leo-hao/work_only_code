#!/bin/bash
mkdir dataset/cityscapse
mkdir dataset/gta5
mkdir work
tar -zxvf /root/data/user/cityscapes/cityscapes.tar.gz -C /root/dataset/cityscape
tar -zxvf /root/data/user/GTA5/images.tar.gz -C /root/dataset/gta5
tar -zxvf /root/data/user/GTA5/images2.tar.gz -C /root/dataset/gta5
tar -zxvf /root/data/user/GTA5/images3.tar.gz -C /root/dataset/gta5
tar -zxvf /root/data/user/GTA5/labels.tar.gz -C /root/dataset/gta5
tar -zxvf /root/data/user/GTA5/others.tar.gz -C /root/dataset/gta5

unzip /root/data/user/work/work.zip -d /root/work