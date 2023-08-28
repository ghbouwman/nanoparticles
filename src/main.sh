#!/usr/bin/env sh

mkdir ../plots
mkdir ../utils
mkdir ../figures

rm ../figures/*.png
./main.py

echo "Creating GIFs (this might take a while...)"

convert -delay 5 ../figures/current*.png ../plots/current_WIP.gif
mv ../plots/current_WIP.gif ../plots/current.gif
echo "Finished creating current GIF"

convert -delay 5 ../figures/voltage*.png ../plots/voltage_WIP.gif
mv ../plots/voltage_WIP.gif ../plots/voltage.gif
echo "Finished creating voltage GIF"
