# Research Notes

This file keeps project-relevant notes from supporting papers. It is not a full literature review.

The current project uses these papers as background for on-chip meta-holography, multiplexed channels, phase/amplitude control, and image-quality goals. The implementation is still an FFT proxy optimizer.

## Local PDFs

PDFs are stored in `docs/supporting_data_pdf/`:

- `On-chip meta-optics for semi-transparent screen display in sync with AR projection.pdf`
- `Augmented Reality Enabled by On‐Chip Meta‐Holography Multiplexing.pdf`
- `wang-et-al-2024-compression-encrypted-meta-optics-for-storage-efficiency-and-security-enhancement.pdf`
- `Direct-Printing Hydrogel-Based Platform.pdf`

## On-chip meta-optics for semi-transparent screen display in sync with AR projection

Public source:

- [Optica article page](https://opg.optica.org/abstract.cfm?uri=optica-9-6-670)

Project-relevant points:

- Demonstrates on-chip metasurface integration on a waveguide for synchronized display and AR holography.
- Uses guided-wave incidence and meta-diatom displacement/interference to control local scattering intensity.
- Uses detour phase for holographic projection channels.
- Supports the idea that both intensity/brightness control and phase control matter.

How it maps to this project:

- Current `phdx/phdy` are proxy phase maps, not direct nanostructure geometry.
- The current project does not yet model meta-diatom displacement, scattering efficiency, or fabrication geometry.
- The paper supports future separation of phase control and brightness/efficiency control.

## Augmented Reality Enabled by On-Chip Meta-Holography Multiplexing

Public source:

- [DOI page](https://doi.org/10.1002/lpor.202100638)

Project-relevant points:

- Demonstrates multiplexed on-chip meta-holography for AR.
- Uses independent encoding freedom across multiple channels.
- Hybridizes detour phase and geometric/Pancharatnam-Berry phase ideas.
- Shows why channel multiplexing is a central design concern.

How it maps to this project:

- Current 9-channel optimization is an algorithmic proxy for multiplexed channel control.
- The current code optimizes channel images, not physical meta-atom layouts.
- Future work should keep channel independence and energy balance visible in diagnostics.

## Compression-Encrypted Meta-Optics for Storage Efficiency and Security Enhancement

Public source:

- [ACS Photonics article page](https://pubs.acs.org/doi/10.1021/acsphotonics.3c01519)

Project-relevant points:

- Focuses on optical encryption, compression, and information storage using metasurface channels.
- Shows that target encoding can strongly affect storage efficiency and reconstruction.
- Useful as a reminder that target representation matters, not only the optimizer.

How it maps to this project:

- This is not the main physical design target.
- It may inspire future target encoding/compression ideas if direct grayscale targets remain too hard.
- Do not treat encryption/storage goals as current project goals.

## Direct-Printing Hydrogel-Based Platform

Public source:

- [Publication listing](https://faculty.yangtzeu.edu.cn/daichenjie/en/lwcg/209049/content/28882.htm)

Project-relevant points:

- Related to dynamic full-color printing and holography.
- More relevant as broad display/holography background than as a direct model for the current FFT pipeline.

How it maps to this project:

- Keep as background reference.
- Do not use it to justify changes to the current on-chip FFT proxy model without a specific user request.

## Current Research Boundary

Use the papers to inform:

- channel multiplexing language
- phase-control concepts
- brightness/efficiency constraints
- why target encoding matters

Do not claim the current code implements:

- real meta-diatom geometry
- RCWA/FDTD validation
- fabrication-aware unit-cell mapping
- physical hydrogel printing behavior
- optical encryption workflows
