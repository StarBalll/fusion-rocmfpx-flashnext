# Fusion ROCm build environment
# Base: kyuz0's Strix Halo toolboxes (ROCm 7.x userland, gfx1151 ready)
FROM kyuz0/amd-strix-halo-toolboxes:latest

# Persist the install prefix across container restarts (mount a volume here)
VOLUME ["/opt/llama-rocmfpx-next"]

# Build outputs land here when /work is mounted as the repo checkout
WORKDIR /work

CMD ["./fusion/build.sh", "--skip-patches"]
