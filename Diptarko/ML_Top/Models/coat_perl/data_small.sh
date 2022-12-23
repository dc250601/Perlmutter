#! /bin/bash
echo "Starting the data extraction process..."
cp /pscratch/sd/t/train130/Top_small.zip /dev/shm/data.zip
unzip -q /dev/shm/data.zip -d /dev/shm/
mv /dev/shm/Top_small /dev/shm/Data_small
echo "Done"
