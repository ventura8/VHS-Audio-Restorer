# Project Overview

The **AI-Hybrid VHS Audio Restorer** is a specialized tool optimized for restoring low-fidelity VHS audio using a multi-stage AI pipeline. It is designed to run on high-end hardware (e.g., RTX 5090) but includes auto-tuning for lower-spec configurations.

# Directory Structure

- `input/`: Source video files (MP4, MKV, etc.).
- `output/`: Final restored videos.
- `temp_work/`: Temporary directory for intermediate tracks (automatically purged on success or skip).
- `venv/`: Local Python virtual environment.
- `assets/`: UI assets like logos.
