# AI Instructions: AI-Hybrid VHS Audio Restorer

This document provides technical guidance for AI agents and developers working on this project.

## Core Directives

### 1. Smart Fixes: Lint & Test Pass
When fixing issues in a file, follow this order of operations in a single pass:
- **Lint First**: Run and fix all linting problems (`flake8`, `mypy`) before addressing tests. This ensures the code is clean and syntactically correct before logic is tested.
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

### 4. Pipeline Architecture (Lossless Background)
- **Separation Strategy**: The project uses a **Subtractive/Lossless Background** approach.
    - **Do NOT** use a dedicated "Music" model (like MDX-NET) as it filters out ambient sounds (birds, nature).
    - **DO** use `BS-Roformer` to extract "Vocals". The "Background" (Instrumental) stem from this process is used as the backing track. This guarantees 100% retention of non-vocal audio.
- **Parallelization**: 
    - Enhancement (Vocals) and Denoising (Background) are independent processes.
    - Ensure code supports running these in **parallel threads/processes** to maximize GPU utilization on supported hardware (ULTRA/EXTREME profiles).

## Documentation Index

- [Project Overview & Directory Structure](../docs/project_overview.md)
- [Key Logic & Pipeline](../docs/pipeline_logic.md)
- [Hardware Optimization](../docs/hardware_optimization.md)
- [Configuration](../docs/configuration.md)
- [Development & Standards](../docs/development_standards.md)
