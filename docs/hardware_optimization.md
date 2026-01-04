# Hardware Optimization

- **Auto-Detection**: The `get_optimal_settings()` function detects CPU cores and GPU VRAM.
- **Profiles**:
    -   `EXTREME` (24GB+ VRAM): Batch size 32. **Parallel Processing Enabled**.
    -   `ULTRA` (22GB+ VRAM): Batch size 16. **Parallel Processing Enabled**.
    -   `HIGH` (15GB+ VRAM): Batch size 8.
    -   `MID` (10GB+ VRAM): Batch size 4.
    -   `LOW`: Batch size 1.
- **OOM Resiliency**: `attempt_run_with_retry` and `attempt_cpu_run_with_retry` dynamically reduce batch sizes/threads on failure.
