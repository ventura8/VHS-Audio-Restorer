# AI Instructions: AI-Hybrid VHS Audio Restorer

This document provides technical guidance for AI agents and developers working on this project.

## Core Directives

### 1. Smart Fixes: Lint & Test Pass
When fixing issues in a file, follow this order of operations in a single pass:
- **Lint First**: Run `flake8` to identify issues.
- **Auto-Fix**: ALWAYS run `autopep8 --in-place --aggressive --recursive <module/file>` to automatically resolve style and formatting issues.
- **Manual Fix**: If lints remain after autopep8, fix them manually before addressing tests.
- **Tests Second**: Once lints pass, run tests (`pytest`).
- **Single Pass**: Aim to resolve both lint and test issues in the same iteration whenever possible.

### 2. Coverage & Badges
- **Minimum Coverage**: Maintain at least **90%** code coverage.
- **Badge Generation**: Always generate/update the coverage badge locally after running tests. 
- **Verification**: Check the generated badge to ensure it reflects $\ge 90\%$. If coverage drops below 90%, add tests or optimize code to meet the requirement.

### 3. Cross-Platform Mocking
- **Windows/Linux Compatibility**: Always use mocks and tests that are compatible with both Windows and Linux.
- **Dynamic Attributes**: When mocking platform-specific operations (e.g., `os.add_dll_directory` or `ctypes.windll`), use `autospec=True` or `create=True` in `unittest.mock.patch` to avoid `AttributeError` on systems where those attributes don't exist.
- **Path Handling**: Use `pathlib` for all file path operations to ensure cross-platform compatibility.

### 4. Robust Resume & File Validity
- **Check Validity, Not Just Existence**: When checking if a step is done, never rely solely on `path.exists()`. Always assume a file might be a 0-byte corruption from a crash.
    - Use helpers like `is_valid_audio(path)` or `is_valid_video(path)`.
- **Skip Logic**: Ensure every potentially expensive step has a "Skip if Exists & Valid" check at the very top.

### 5. Pipeline Architecture (Lossless Background)
- **Separation Strategy**: The project uses a **Subtractive/Lossless Background** approach.
    - **Do NOT** use a dedicated "Music" model (like MDX-NET) as it filters out ambient sounds (birds, nature).
    - **DO** use `BS-Roformer` to extract "Vocals". The "Background" (Instrumental) stem from this process is used as the backing track. This guarantees 100% retention of non-vocal audio.
- **Parallelization**: 
    - **Stability Priority**: While the engine supports parallel threads, the *high-level* pipeline defaults to **Sequential** execution for Steps 2-4.
    - **Reason**: Running heavy GPU/CPU tasks in parallel on Windows causes `stdout` contention (interleaved progress bars) and potential race conditions. 
    - **Note**: Development should favor UI stability (clean, readable logs) over raw theoretical speed if they conflict.

### 6. Synchronization Logic
- **Methods**: The project supports two sync methods:
    - `shift` (Default): Global delay correction (Cross-Correlation). Fast and artifact-free.
    - `dtw`: Dynamic Time Warping. Corrects variable drift (wow/flutter).
        - **Hybrid Engine**: Uses `torch.cdist` on GPU for distance calculation + CPU for pathfinding.
        - **Precision**: 40Hz - 100Hz.
- **Implementation**: Sync logic is in `_align_stems` / `_align_stems_dtw`.
    - **UI Standards**: Every task with a progress bar MUST explicitly call `draw_progress_bar(100, ...)` before completion.
    - **Sequential Execution**: Syncing Vocals and Music runs sequentially to maintain a clean grouped log output.
    - **Dynamic Radius**: `radius` for DTW must scale with resolution (e.g., `0.3 * Resolution`).
    - **Recommended Resolution**: 100Hz for high precision (lipsync), 40Hz for general speed correction.

### 6. Runtime Patching
- **Mechanism**: The project uses `apply_patches.py` to fix upstream library issues (e.g., `soundfile`, `deepspeed`) during runtime or installation.
- **Maintenance**: Always verify patches still apply correctly after a dependency update (`pip install -U`).

### 7. Execution Modes
- **Interactive**: Running without arguments or double-clicking `start.bat` triggers an interactive prompt for drag-and-drop or scanning the `input` folder.
- **CLI**: Passing a file or directory as a command-line argument processes those targets directly and outputs to the same folder as the input.

## Documentation Index

- [Project Overview & Directory Structure](../docs/project_overview.md)
- [Key Logic & Pipeline](../docs/pipeline_logic.md)
- [Hardware Optimization](../docs/hardware_optimization.md)
- [Configuration](../docs/configuration.md)
- [Development & Standards](../docs/development_standards.md)
