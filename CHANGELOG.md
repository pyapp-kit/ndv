# Changelog

## [v0.2.1](https://github.com/pyapp-kit/ndv/tree/v0.2.1) (2025-01-17)

[Full Changelog](https://github.com/pyapp-kit/ndv/compare/v0.2.0...v0.2.1)

**Merged pull requests:**

- chore: update dependencies [\#104](https://github.com/pyapp-kit/ndv/pull/104) ([tlambert03](https://github.com/tlambert03))

## [v0.2.0](https://github.com/pyapp-kit/ndv/tree/v0.2.0) (2025-01-17)

[Full Changelog](https://github.com/pyapp-kit/ndv/compare/v0.0.4...v0.2.0)

**Implemented enhancements:**

- feat: add reset zoom button to jupyter [\#102](https://github.com/pyapp-kit/ndv/pull/102) ([tlambert03](https://github.com/tlambert03))
- feat: adding back async [\#92](https://github.com/pyapp-kit/ndv/pull/92) ([tlambert03](https://github.com/tlambert03))
- feat: support interactive usage with wx on ipython [\#89](https://github.com/pyapp-kit/ndv/pull/89) ([tlambert03](https://github.com/tlambert03))
- feat: return 3d support to v2 [\#83](https://github.com/pyapp-kit/ndv/pull/83) ([tlambert03](https://github.com/tlambert03))
- feat: support GPU-calculated luts in vispy [\#77](https://github.com/pyapp-kit/ndv/pull/77) ([tlambert03](https://github.com/tlambert03))
- feat: add CanvasProvider and GuiProvider classes [\#75](https://github.com/pyapp-kit/ndv/pull/75) ([tlambert03](https://github.com/tlambert03))
- feat: better excepthook in v2 [\#69](https://github.com/pyapp-kit/ndv/pull/69) ([tlambert03](https://github.com/tlambert03))
- feat: v2 histogram \(simplified alternative\) [\#65](https://github.com/pyapp-kit/ndv/pull/65) ([tlambert03](https://github.com/tlambert03))
- feat: add Wx front-end [\#62](https://github.com/pyapp-kit/ndv/pull/62) ([tlambert03](https://github.com/tlambert03))
- feat: V2 add back channel modes and multi-channel display [\#57](https://github.com/pyapp-kit/ndv/pull/57) ([tlambert03](https://github.com/tlambert03))
- feat: Histogram Widget [\#52](https://github.com/pyapp-kit/ndv/pull/52) ([gselzer](https://github.com/gselzer))
- feat: Use handles to compute viewer range [\#38](https://github.com/pyapp-kit/ndv/pull/38) ([gselzer](https://github.com/gselzer))
- feat: better mouse events and canvas2world methods [\#26](https://github.com/pyapp-kit/ndv/pull/26) ([tlambert03](https://github.com/tlambert03))
- feat: Add ROI selection tool [\#23](https://github.com/pyapp-kit/ndv/pull/23) ([gselzer](https://github.com/gselzer))

**Fixed bugs:**

- fix: remove texture guessing for vispy, try texture\_format='auto' before falling back [\#98](https://github.com/pyapp-kit/ndv/pull/98) ([tlambert03](https://github.com/tlambert03))
- fix: make sure it's ok to set\_data to `None` and generally have an empty viewer [\#43](https://github.com/pyapp-kit/ndv/pull/43) ([tlambert03](https://github.com/tlambert03))
- fix: fix canvas update on set\_data [\#40](https://github.com/pyapp-kit/ndv/pull/40) ([tlambert03](https://github.com/tlambert03))
- fix: Clean up PyGFX events [\#34](https://github.com/pyapp-kit/ndv/pull/34) ([gselzer](https://github.com/gselzer))
- Fix the height of the info label [\#32](https://github.com/pyapp-kit/ndv/pull/32) ([hanjinliu](https://github.com/hanjinliu))
- fix: fix slider signals for pyside6 [\#31](https://github.com/pyapp-kit/ndv/pull/31) ([tlambert03](https://github.com/tlambert03))

**Tests & CI:**

- test: Add back tests [\#74](https://github.com/pyapp-kit/ndv/pull/74) ([tlambert03](https://github.com/tlambert03))
- ci: test lots of backends on ci [\#73](https://github.com/pyapp-kit/ndv/pull/73) ([tlambert03](https://github.com/tlambert03))
- test: add test for controller, with mocks for canvas and frontend objects [\#72](https://github.com/pyapp-kit/ndv/pull/72) ([tlambert03](https://github.com/tlambert03))

**Documentation:**

- docs: more updates to API documentation [\#103](https://github.com/pyapp-kit/ndv/pull/103) ([tlambert03](https://github.com/tlambert03))
- docs: minor installation fixes & cleanup [\#101](https://github.com/pyapp-kit/ndv/pull/101) ([gselzer](https://github.com/gselzer))
- docs: update documentation to list all env vars [\#97](https://github.com/pyapp-kit/ndv/pull/97) ([tlambert03](https://github.com/tlambert03))
- docs: more docs work [\#93](https://github.com/pyapp-kit/ndv/pull/93) ([tlambert03](https://github.com/tlambert03))
- docs: add pre-release banner [\#90](https://github.com/pyapp-kit/ndv/pull/90) ([tlambert03](https://github.com/tlambert03))
- docs: add versioned docs with deployment [\#85](https://github.com/pyapp-kit/ndv/pull/85) ([tlambert03](https://github.com/tlambert03))
- docs: setup documentation with mkdocs [\#84](https://github.com/pyapp-kit/ndv/pull/84) ([tlambert03](https://github.com/tlambert03))

**Merged pull requests:**

- refactor: remove `LUTModel.autoscale` and make `LUTModel.clims` more powerful [\#94](https://github.com/pyapp-kit/ndv/pull/94) ([tlambert03](https://github.com/tlambert03))
- refactor: Better Styles for wx and jupyter widgets [\#88](https://github.com/pyapp-kit/ndv/pull/88) ([tlambert03](https://github.com/tlambert03))
- refactor: merge in v2 "MVC" branch [\#82](https://github.com/pyapp-kit/ndv/pull/82) ([tlambert03](https://github.com/tlambert03))
- refactor: cleanup on public API [\#80](https://github.com/pyapp-kit/ndv/pull/80) ([tlambert03](https://github.com/tlambert03))
- refactor: v2 - use base classes rather than protocols [\#66](https://github.com/pyapp-kit/ndv/pull/66) ([tlambert03](https://github.com/tlambert03))
- refactor: V2 "mouseable" canvas objects, remove mouse logic from qt/jupyter views [\#64](https://github.com/pyapp-kit/ndv/pull/64) ([tlambert03](https://github.com/tlambert03))
- refactor: minor rearrangement of v2 stuff [\#63](https://github.com/pyapp-kit/ndv/pull/63) ([tlambert03](https://github.com/tlambert03))
- refactor: updates to v2-mvc branch [\#56](https://github.com/pyapp-kit/ndv/pull/56) ([tlambert03](https://github.com/tlambert03))
- chore: autoupdate pre-commit, drop python 3.8 [\#37](https://github.com/pyapp-kit/ndv/pull/37) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))

## [v0.0.4](https://github.com/pyapp-kit/ndv/tree/v0.0.4) (2024-06-12)

[Full Changelog](https://github.com/pyapp-kit/ndv/compare/v0.0.3...v0.0.4)

**Merged pull requests:**

- fix: tensorstore isel [\#25](https://github.com/pyapp-kit/ndv/pull/25) ([tlambert03](https://github.com/tlambert03))
- docs: add minimal pygfx and vispy example [\#24](https://github.com/pyapp-kit/ndv/pull/24) ([tlambert03](https://github.com/tlambert03))

## [v0.0.3](https://github.com/pyapp-kit/ndv/tree/v0.0.3) (2024-06-09)

[Full Changelog](https://github.com/pyapp-kit/ndv/compare/v0.0.2...v0.0.3)

**Merged pull requests:**

- build: add back support for python3.8 [\#19](https://github.com/pyapp-kit/ndv/pull/19) ([tlambert03](https://github.com/tlambert03))
- fix: fix progress spinner on first show [\#18](https://github.com/pyapp-kit/ndv/pull/18) ([tlambert03](https://github.com/tlambert03))
- Update README.md [\#16](https://github.com/pyapp-kit/ndv/pull/16) ([tlambert03](https://github.com/tlambert03))
- feat: enable 3d for pygfx [\#15](https://github.com/pyapp-kit/ndv/pull/15) ([tlambert03](https://github.com/tlambert03))
- refactor: minimize public API surface [\#13](https://github.com/pyapp-kit/ndv/pull/13) ([tlambert03](https://github.com/tlambert03))
- feat: add spinner when loading frames [\#9](https://github.com/pyapp-kit/ndv/pull/9) ([tlambert03](https://github.com/tlambert03))

## [v0.0.2](https://github.com/pyapp-kit/ndv/tree/v0.0.2) (2024-06-08)

[Full Changelog](https://github.com/pyapp-kit/ndv/compare/v0.0.1...v0.0.2)

**Merged pull requests:**

- support more array types, add `imshow` [\#3](https://github.com/pyapp-kit/ndv/pull/3) ([tlambert03](https://github.com/tlambert03))
- move repo [\#2](https://github.com/pyapp-kit/ndv/pull/2) ([tlambert03](https://github.com/tlambert03))

## [v0.0.1](https://github.com/pyapp-kit/ndv/tree/v0.0.1) (2024-06-07)

[Full Changelog](https://github.com/pyapp-kit/ndv/compare/97edb3b0f787b6a19ab965afb40475b13d275795...v0.0.1)

**Merged pull requests:**

- ci\(dependabot\): bump softprops/action-gh-release from 1 to 2 [\#1](https://github.com/pyapp-kit/ndv/pull/1) ([dependabot[bot]](https://github.com/apps/dependabot))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
