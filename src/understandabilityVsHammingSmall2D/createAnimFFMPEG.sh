#!/bin/bash

# ---- Define your parameters here ----
L=256
gamma=1
alpha=1
r=1
B=16
mu=0.001
K=3

framerate=20  # Frames per second for the animation
frame_start_num=0
# -------------------------------------

# Compose the directory name
DIRNAME="L_${L}_g_${gamma}_a_${alpha}_r_${r}_B_${B}_mu_${mu}_K_${K}"
FRAMES_DIR="src/understandabilityVsHammingSmall2D/plots/latticeAnimNbrVsGlobal/frames/$DIRNAME"
OUTDIR="src/understandabilityVsHammingSmall2D/plots/latticeAnimNbrVsGlobal"
OUTFILE="$OUTDIR/${DIRNAME%/}.mp4"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "Frames directory not found: $FRAMES_DIR"
    exit 2
fi

ffmpeg -y -framerate $framerate -start_number $frame_start_num \
  -i "$FRAMES_DIR/frame_%04d.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p "$OUTFILE"


# # target ~80-90MB
# ffmpeg -y -framerate $framerate -start_number $frame_start_num \
#   -i "$FRAMES_DIR/frame_%04d.png" \
#   -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
#   -c:v libx264 -b:v 15000k -maxrate 20000k -bufsize 40000k \
#   -pix_fmt yuv420p "$OUTFILE"

echo "Saved animation to $OUTFILE"