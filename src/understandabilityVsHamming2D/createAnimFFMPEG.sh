#!/bin/bash

# ---- Define your parameters here ----
L=128
gamma=2
alpha=1
r=0.5
B=32
mu=0.01
K=3

framerate=10  # Frames per second for the animation
frame_start_num=900
# -------------------------------------

# Compose the directory name
DIRNAME="L_${L}_g_${gamma}_a_${alpha}_r_${r}_B_${B}_mu_${mu}_K_${K}"
FRAMES_DIR="src/understandabilityVsHamming2D/plots/latticeAnimNbrVsGlobal/frames/$DIRNAME"
OUTDIR="src/understandabilityVsHamming2D/plots/latticeAnimNbrVsGlobal"
OUTFILE="$OUTDIR/${DIRNAME%/}.mp4"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "Frames directory not found: $FRAMES_DIR"
    exit 2
fi

ffmpeg -y -framerate $framerate -start_number $frame_start_num \
  -i "$FRAMES_DIR/frame_%04d.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p "$OUTFILE"

echo "Saved animation to $OUTFILE"