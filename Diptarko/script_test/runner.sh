#!/bin/bash
echo "Script Started"

start=$(($(date +%S)+$(($(date +%M)*60))))

timeout 1m python3 pause.py
end=$(($(date +%S)+$(($(date +%M)*60))))

tot=$((end-start))
if [ $tot > 60 ]; then
	echo "Number is Even"
fi
#bash runner.sh
