#!/bin/bash

# ---- Define your parameters here ----
framerate=10  # Frames per second for the animation
frame_start_num=900
B=4  # Bit string length parameter
# -------------------------------------

# Compose the directory name
FRAMES_DIR="src/understandabilityVsHammingSmall/plots/languages/frames/"
OUTDIR="src/understandabilityVsHammingSmall/plots/languages"
OUTFILE="$OUTDIR/surviving_languages_heatmap_B_${B}.mp4"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "Frames directory not found: $FRAMES_DIR"
    exit 2
fi

ffmpeg -y -framerate $framerate -start_number $frame_start_num \
  -i "$FRAMES_DIR/frame_%04d.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p "$OUTFILE"

echo "Saved animation to $OUTFILE"