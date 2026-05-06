# Changelog

All notable changes to sandlerchemeq will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Version bumped to 0.3.2; 0.3.1 was already published to PyPI under a prior tag and could not be reused.

## [0.3.1] - 2026-05-06

### Fixed

- `ChemEqSystem.solve_lagrange`: `fsolve` call was inside the `if len(zGuess) == 0` branch, causing a `NameError` when a non-empty `zInit` was supplied.
- `ChemEqSystem.solve_implicit`: `Xinit` default changed from `[]` to `None`; a `None` value (as passed by the CLI when `--X-init` is omitted) is now treated as zeros, preventing a crash.
- CLI `--pressure` help text corrected from "MPa" to "bar" to match the unit assumed by the code.
- `test_component_inheritance_from_compound`: updated `T`, `P`, and `dGf_T` assertions to use pint-aware comparisons (`.m_as()`).

## [0.3.0] - 2026-02-21

### Added

- Full pint-unit support throughout: `T`, `P`, `Tref` stored as `pint.Quantity`; `R`, `RT`, `dGr`, `dHr` likewise.
- `Component.dGf_T` returns `pint.Quantity` in J/mol.
- `ChemEqSystem.RT` attribute (`pint.Quantity`, J/mol).

### Fixed

- `ChemEqSystem.report` was defined at module scope instead of as a class method; corrected indentation.

## [0.2.1] - 2026-01-06

### Added

- `roman` dependency.

## [0.2.0] - 2026-01-06

### Added

- Full van't Hoff option available via `simplified=False` argument to `ChemEqSystem.solve_implicit`.

## [0.1.0] - 2026-01-01

### Added

- Initial release.
- Lagrange multiplier method (`ChemEqSystem.solve_lagrange`).
- Explicit reaction method (`ChemEqSystem.solve_implicit`).
- Command-line interface `sandlerchemeq solve`.