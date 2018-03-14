#!/bin/bash

for i in *.avi ; do
	mkdir ${i/.avi}
	ffmpeg -i $i ${i/.avi}/${i/.avi}_%03d.png
done
