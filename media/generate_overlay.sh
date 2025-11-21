#!/bin/bash

python media/generate_frames.py

ffmpeg -framerate 50 -i media/frames/frame_%06d.png \
       -c:v prores_ks -pix_fmt yuva444p10le \
       media/overlay.mov -y

ffmpeg -i logs/run_26-episode-0.mp4 -i media/overlay.mov \
       -filter_complex "[0:v][1:v] overlay=0:0" \
       -c:v libx264 -pix_fmt yuv420p \
       media/final_video.mp4 -y