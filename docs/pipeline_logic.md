# Key Logic & Pipeline

The script `restore_audio_hybrid.py` follows a 5-step restoration flow:

1.  **Extraction**: FFmpeg extracts audio as `pcm_f32le` (32-bit float) to prevent clipping.
2.  **Separation (Hybrid)**:
    -   **Vocals**: BS-Roformer-Viperx-1297 (best for vocal isolation).
    -   **Music**: UVR-MDX-NET-Inst_HQ_3 (cleanest instrumental separation).
3.  **Vocal Enhancement**: Resemble-Enhance (Denoise Only) performs high-quality cleaning.
4.  **Music Denoising**: UVR-DeNoise-Lite cleans the instrumental track.
5.  **Smart Audio Sync**: Calculates the sample-accurate latency (cross-correlation) between the original and processed audio, correcting any drift or processing delays.
6.  **Final Mix**: FFmpeg recombines the aligned stems into a 32-bit float stereo mix with the original video.
