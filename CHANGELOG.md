# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [1.0.2] - 2021-10-26

### Added

- Added references to a separate [experiment repository](https://github.com/fsschneider/cockpit-experiments) that publishes the code for all experiments shown in the paper.

### Fixed

- Protects the `batch_grad` field in the case where non-SGD is used together with other quantities that free `batch_grad` for memory performance. [[#5](https://github.com/f-dangel/cockpit/issues/5), [PR](https://github.com/f-dangel/cockpit/pull/18)]

## [1.0.1] - 2021-10-13

From this version on, `cockpit` will be available as `cockpit-for-pytorch` on
PyPI.

### Added
- Make library `pip`-installable as `cockpit-for-pytorch`
  [[PR](https://github.com/f-dangel/cockpit/pull/17)]
- Require BackPACK main release
  [[PR](https://github.com/f-dangel/cockpit/pull/12)]
- Added a `savename` argument to the `CockpitPlotter.plot()` function, which lets you define the name, and now the `savedir` should really only describe the **directory**. [[PR](https://github.com/f-dangel/cockpit/pull/16), Fixes #8]
- Added optional `savefig_kwargs` argument to the `CockpitPlotter.plot()` function that gets passed to the `matplotlib` function `fig.savefig()` to, e.g., specify DPI value or use a different file format (e.g. PDF). [[PR](https://github.com/f-dangel/cockpit/pull/16), Fixes #10]

### Internal
- Fix [#6](https://github.com/f-dangel/cockpit/issues/6): Don't execute
  extension hook on modules with children
  [[PR](https://github.com/f-dangel/cockpit/pull/7)]

## [1.0.0] - 2021-04-30

### Added

- First public release version of **Cockpit**.

[Unreleased]: https://github.com/f-dangel/cockpit/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/f-dangel/cockpit/compare/1.0.1...1.0.2
[1.0.1]: https://github.com/f-dangel/cockpit/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/f-dangel/cockpit/releases/tag/1.0.0
