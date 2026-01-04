---
description: Fix lint and test issues for a file in a single pass, ensuring 90% coverage and cross-platform compatibility.
---

// turbo-all
# Workflow: Fix Lints and Tests

This workflow ensures that code quality and testing standards are met in a single, efficient pass.

### 1. Run Lints
First, identify all linting issues for the target file.
```powershell
flake8 <file_path>
mypy <file_path>
```

### 2. Fix Lints
Apply fixes for all reported linting errors. Prioritize fixing lints before moving to tests.

### 3. Run Tests & Coverage
Once lints pass, run tests and check coverage.
```powershell
pytest --cov=<module_name> --cov-report=term-missing <test_path>
```

### 4. Fix Test Failures
Resolve any failing tests. 
- Ensure mocks are **Windows/Linux compatible** (use `autospec=True` or `create=True` for platform-specific attributes).
- Use `pathlib` for any path-related fixes.

### 5. Generate Badge & Verify
Generate the coverage badge and verify it meets the 90% threshold.
```powershell
# Example command if using a specific script or coverage-badge tool
# python .github/scripts/generate_coverage_badge.py
coverage-badge -o assets/coverage.svg -f
```
> [!IMPORTANT]
> Always manually check that the coverage is at least **90%**.
