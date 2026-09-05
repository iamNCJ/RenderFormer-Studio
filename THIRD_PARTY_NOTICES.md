# Third-party notices

RenderFormer-authored code is licensed under the repository's MIT license.
The files listed below retain their upstream licenses and copyright notices.
Pinned commits identify the source reviewed for this release; links to a paper
or project are not substitutes for these software licenses.

## Microsoft RenderFormer V1 release

- Local scope: V1-compatible portions of `src/renderformer/models/`,
  `src/renderformer/pipeline.py`, `src/renderformer/data/compat/`,
  `src/renderformer/data/geometry/camera.py`,
  `src/renderformer/data/pipelines/compose_rf1.py`, and unchanged official
  example/template files identified by their asset manifests
- Source: `microsoft/renderformer` at commit
  `c51f87083d0eebb806803fe32a357b8d9aefb085`
- Upstream: <https://github.com/microsoft/renderformer>
- Copyright: Microsoft Corporation
- License: MIT, reproduced in
  `licenses/microsoft-renderformer-MIT.txt`
- Modifications: the V1 model and scene/inference paths were merged with the
  training implementation and the V2 pipeline; compatibility aliases and
  exact legacy preprocessing were retained.

## DPT blocks

- Local scope: `src/renderformer/models/dpt.py`
- Source: `isl-org/DPT` at commit
  `cd3fe90bb4c48577535cc4d51b602acca688a2ee`, primarily `dpt/blocks.py`
- Upstream: <https://github.com/isl-org/DPT>
- Copyright: 2021 Intel ISL (Intel Intelligent Systems Lab)
- License: MIT, reproduced in `licenses/DPT-MIT.txt`
- Modifications: retained block helpers and added the RenderFormer decoder
  integration.

## Rotary embedding

- Local scope: `src/renderformer/models/rope.py`
- Source: `lucidrains/rotary-embedding-torch` at commit
  `e2224b5102a045998b4131bcac52c9ebde779b5d`
- Upstream: <https://github.com/lucidrains/rotary-embedding-torch>
- Copyright: 2021 Phil Wang
- License: MIT, reproduced in
  `licenses/rotary-embedding-torch-MIT.txt`
- Modifications: RenderFormer frequency layouts, cached forms, and triangle
  encoding support were added to the upstream implementation.

## Spatial serialization and ordering

- Local scope: `src/renderformer/models/ordering/`
- Immediate source: `Pointcept/PointTransformerV3` at commit
  `37a3ddc031de127240ed558a450f6e684647ebe1`
- Upstream: <https://github.com/Pointcept/PointTransformerV3>
- Copyright: 2023 Pointcept
- License: MIT, reproduced in `licenses/PointTransformerV3-MIT.txt`
- Modifications: batched masks, prefix ordering, shuffle/shift variants, and an
  optional compiled Hilbert path were added.

The PointTransformerV3 Hilbert implementation is a PyTorch port of
`PrincetonLIPS/numpy-hilbert-curve` at commit
`e8712a74cb22953b341f9b3e0a275145c0d19102` (Copyright 2020 Princeton
Laboratory for Intelligent Probabilistic Systems, MIT). Its license is in
`licenses/numpy-hilbert-curve-MIT.txt`.

The Z-order implementation traces through `octree-nn/ocnn-pytorch` at commit
`f9fbffdd36c51e75afc61349a5d9943d82c38ccd` (Copyright 2022 Peng-Shuai Wang,
MIT). Its license is in `licenses/ocnn-pytorch-MIT.txt`.

## Nerfstudio encoding

- Local scope: `src/renderformer/models/pe.py`
- Source: `nerfstudio-project/nerfstudio` at commit
  `f0aadf9de8b9da4c576a207ad399450bbac4315e`
- Upstream: <https://github.com/nerfstudio-project/nerfstudio>
- Copyright: 2022 The Nerfstudio Team
- License: Apache License 2.0, reproduced in `licenses/Apache-2.0.txt`
- Modifications: the encoding subset was adapted for RenderFormer tensors.

## Liger Kernel and Unsloth portions

- Local scope: `src/renderformer/ops/`
- Source: `linkedin/Liger-Kernel` at commit
  `7382a8761f9af679482b968f9348013d933947c7`
- Upstream: <https://github.com/linkedin/Liger-Kernel>
- Copyright: 2024 LinkedIn Corporation
- License: BSD 2-Clause, reproduced in `src/renderformer/ops/LICENSE.txt`; the
  upstream notice is reproduced in `src/renderformer/ops/NOTICE`
- Modifications: kernels were selected and integrated with RenderFormer's
  replacement layer.

The indicated portions of `src/renderformer/ops/_triton/rms_norm.py` and
`src/renderformer/ops/_triton/utils.py` incorporate Unsloth code from commit
`fd753fed99ed5f10ef8a9b7139588d9de9ddecfb`.

- Upstream: <https://github.com/unslothai/unsloth>
- Copyright: 2023-present Daniel Han-Chen and the Unsloth team
- License: Apache License 2.0, reproduced in `licenses/Apache-2.0.txt`
- Modifications: the portions were incorporated through Liger Kernel and
  adapted for its Triton operations.

## Dependencies, models, datasets, and assets

Package dependencies and optional model/data downloads are not redistributed
under the repository's MIT license. Users must follow each provider's terms.
Example and training assets are governed by their provenance manifests; an
asset is distributable only when its `redistribution_status` is `approved`.

The bundled RF1 example OBJ files are mapped individually in
`examples/rf1/ASSET_PROVENANCE.json` and credited in
`examples/rf1/ASSET_ATTRIBUTIONS.md`. This includes CC BY 4.0, CC0 1.0,
Stanford repository terms, and source-specific Cornell, Utah, and Mitsuba
notices. Full CC BY 4.0 and CC0 1.0 legal texts are reproduced in `licenses/`.
Bundled RF2 prepared-frame sources and paper reference images are credited in
`examples/rf2/PAPER_SCENE_ATTRIBUTIONS.md`. Generated RF2 H5 files and model
weights are not stored in the repository; inputs whose redistribution rights
could not be established must be supplied locally as documented there.
Project-authored template OBJ files are mapped in
`src/renderformer/data/templates/ASSET_PROVENANCE.json`.
