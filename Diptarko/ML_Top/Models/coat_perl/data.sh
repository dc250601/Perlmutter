#! /bin/bash
echo "Starting the data extraction process..."
cp /pscratch/sd/t/train130/Top.zip /dev/shm/data.zip
unzip -q /dev/shm/data.zip -d /dev/shm/
echo "Done"
