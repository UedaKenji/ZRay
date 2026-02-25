# Changelog

## [0.1.3] - 2026-02-25
### Added
- Added `MANIFEST.in` and setuptools package-data settings to include bundled DXF files in built distributions.
- Added development dependency group and basic `uv` workflow notes.

### Changed
- Improved packaging metadata and dependencies in `pyproject.toml`.
- Updated README usage/setup notes and corrected axisymmetric spelling.

### Fixed
- Fixed plotting helpers to support externally provided figures/axes more robustly.
- Fixed `LineCamera.sensor_size` normalization and removed debug prints.
- Fixed `Camera2D_rphiz` coordinate conversion and 3D vessel axis aspect calculation compatibility.

## [0.1.2] - 2025-05-09
### Removed
- Removed OpenCV dependency by replacing `cv2.floodFill` with a custom `flood_fill_numpy` function.

### Changed
- Updated version to 0.1.2 in `__init__.py`.

## [0.1.1] - 2025-05-09
### Added
- Added docstring comments to all classes and functions in `main.py` and `measurement.py` for better documentation.

### Changed
- Updated `measurement.py` with additional explanations for functions and classes.

### Fixed
- Minor fixes to ensure consistency in docstring formatting.
