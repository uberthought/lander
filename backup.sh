#! /bin/bash

# if the backup directory doesn't exist, create it
if [ ! -d "../lander_backup" ]; then
  mkdir ../lander_backup
fi

datetime=$(date +%Y-%m-%d_%H-%M-%S)

# Backup the python, shell, and other files
tar -czvf ../lander_backup/lander_$datetime.tar.gz *.py *.sh
