# Key Logic & Pipeline

The script `restore_audio_hybrid.py` follows a 5-step restoration flow:

1.  **Extraction**: FFmpeg extracts audio as `pcm_f32le` (32-bit float) to prevent clipping.
2.  **Separation (Hybrid)**:
    -   **Vocals**: BS-Roformer-Viperx-1297 (best for vocal isolation).
    -   **Background**: Derived subtractively or named `(Background)` directly from the separator to preserve quality.
3.  **Vocal Enhancement**: Resemble-Enhance (Denoise Only) performs high-quality cleaning.
4.  **Background Denoising**: UVR-DeNoise-Lite cleans the background track.
5.  **Smart Audio Sync**: Aligns the processed audio to the original using one of two methods:
    -   **Global Shift (Cross-Correlation)**: Calculates a single best-fit offset (`ref[t] = proc[t + lag]`). Ideal for digital sources or tapes with constant speed.
    -   **Dynamic Time Warping (DTW)**: Hybrid GPU+CPU engine corrects non-linear drift (wow/flutter/speed changes).
6.  **Final Mix**: FFmpeg recombines the aligned stems into a 32-bit float stereo mix with the original video.
