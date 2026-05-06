# Changelog

All notable changes to sandlerchemeq will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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