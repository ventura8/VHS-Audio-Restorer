# Environment & Dependency Management

- **Installation**: `install_dependencies.ps1` sets up a portable environment, including local FFmpeg and PyTorch Nightly (for RTX 50-series support).
- **Patching**: `apply_patches.py` removes DeepSpeed dependencies from Resemble-Enhance and fixes Torchaudio compatibility issues.

# Coding Standards

- **Audio Quality**: Always use 32-bit float (`pcm_f32le`) for intermediate and final audio to maintain maximum dynamic range.
- **Pathing**: Use `pathlib.Path` for cross-platform compatibility.
- **Resiliency**: Always implement `is_valid_audio()` checks before skipping steps to ensure checkpoints are not corrupted.
- **Naming**: Temporary files should include the source stem (e.g., `video_name.wav`).
- **Cleanup**: The `temp_work` directory is automatically purged upon successful task completion.
- **Testing & Coverage**: Maintain a minimum of **90% code coverage** (mandatory CI threshold). Coverage reports (including branch coverage and cyclomatic complexity) are automatically generated in CI and reflected in the job summary and local badges.
- **Badge Mandatory**: Always ensure the `assets/coverage.svg` badge is updated after making code changes. This is handled automatically by local test runs, but must be verified before pushing.
- **Code Quality**: Adhere to `flake8` standards (max line length 127, max complexity 15). Use `mypy` for type checking. Complexity is monitored using `radon`.
- **Badge Automation**: Local `pytest` runs automatically update the `assets/coverage.svg` badge via a `pytest_terminal_summary` hook in `tests/conftest.py`.
- **Documentation**: 
    -   Always update `Instructions.md`, `README.md`, and relevant `docs/` files if necessary when making changes.