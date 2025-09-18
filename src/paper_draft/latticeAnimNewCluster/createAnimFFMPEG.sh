#!/bin/bash

# ---- Define your parameters here ----
L=256
gamma=1
alpha=0.8
B=16
mu=0.0001

framerate=20  # Frames per second for the animation
# -------------------------------------

# Compose the directory name (removed K parameter)
DIRNAME="L_${L}_g_${gamma}_a_${alpha}_B_${B}_mu_${mu}"
FRAMES_DIR="plots/latticeAnim//frames/$DIRNAME"
OUTDIR="plots/latticeAnim/"
OUTFILE="$OUTDIR/${DIRNAME}.mp4"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "Frames directory not found: $FRAMES_DIR"
    exit 2
fi

echo "Creating animation from: $FRAMES_DIR"
echo "Output file: $OUTFILE"

# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"

# ffmpeg -y -framerate $framerate \
#   -pattern_type glob -i "$FRAMES_DIR/frame_*.png" \
#   -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
#   -c:v libx264 -pix_fmt yuv420p "$OUTFILE"

ffmpeg -y -framerate $framerate \
  -pattern_type glob -i "$FRAMES_DIR/frame_*.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p \
  -crf 18 -maxrate 8M -bufsize 12M \
  "$OUTFILE"

echo "Saved animation to $OUTFILE"