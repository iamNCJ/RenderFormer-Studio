# RF1 example asset attributions

The repository's MIT license applies to RenderFormer-authored files. The
third-party meshes below retain their own terms. The authoritative per-file
mapping and checksums are in [`ASSET_PROVENANCE.json`](ASSET_PROVENANCE.json).
Every legacy OBJ in that manifest is byte-for-byte identical to the
corresponding file in
[`microsoft/renderformer@c51f870`](https://github.com/microsoft/renderformer/tree/c51f87083d0eebb806803fe32a357b8d9aefb085/examples).

## CC BY 4.0 assets

The following source models are licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
RenderFormer converted, extracted, split, simplified, and/or remeshed them for
the bundled example scenes; the manifest records the exact local files for
each source.

- [Klein Bottle](https://sketchfab.com/3d-models/klein-bottle-ce95eaceb29544aaa47db7a586811b09)
  by Fausto Javier Da Rosa.
- [Lowpoly Crystals](https://sketchfab.com/3d-models/lowpoly-crystals-d94bccb1305d409482eba04d736fb7dd)
  by Mongze.
- [Lowpoly Fox](https://sketchfab.com/3d-models/lowpoly-fox-01674a892f414c0681afdeb563cc8e13)
  by Vlad Zaichyk.
- [Bronco](https://sketchfab.com/3d-models/bronco-37e3760bfde44beeb39e6fd69b690637)
  by Microsoft.
- [Heart Puzzle](https://sketchfab.com/3d-models/heart-puzzle-e762b2d7de6749e79f54e0e6a0ff96be)
  by Microsoft.
- [Jewelry](https://sketchfab.com/3d-models/jewelry-4373121e41f94727bb802b78ce6b566f)
  by elbenZ.

The Lowpoly Fox upload says its Japanese-environment design used a reference
image by Frost-Skyder. That reference-image right was not independently
verified; this repository includes geometry only.

The full CC BY 4.0 legal code is bundled at
[`licenses/CC-BY-4.0.txt`](../../licenses/CC-BY-4.0.txt).

## CC0 1.0 assets

- [Spot](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/) by
  Keenan Crane.
- [Small Volume Surface of Constant Width](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/),
  described by Andrii Arman, Andriy Bondarenko, Fedor Nazarov, Andriy Prymak,
  and Danylo Radchenko; mesh released by Keenan Crane.
- [Veach MIS](https://d38rqfq1h7iukm.cloudfront.net/scenes/veach-mis.zip) is a
  Mitsuba scene by Benedikt Bitterli, based on the multiple-importance-sampling
  test setup of Eric Veach and Leonidas J. Guibas.

These assets are dedicated under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
The legal code is bundled at
[`licenses/CC0-1.0.txt`](../../licenses/CC0-1.0.txt).

## Repository-specific terms and credits

- Cornell Box geometry is based on the
  [Cornell University Program of Computer Graphics public-use data](https://bowers.cornell.edu/computer-graphics/data).
- Stanford Bunny and Lucy come from the
  [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
  Stanford permits acknowledged research use and free mirroring or
  redistribution, but prohibits commercial use without permission. A local
  terms summary is in
  [`licenses/Stanford-3D-Scanning-Repository-NOTICE.txt`](../../licenses/Stanford-3D-Scanning-Repository-NOTICE.txt).
- The Utah Teapot is based on the public downloads from the
  [University of Utah Model Repository](https://users.cs.utah.edu/~dejohnso/models/teapot.html);
  the original model is by Martin Newell.
- The shader-ball meshes were converted from the
  [Mitsuba material-preview scene](https://www.mitsuba-renderer.org/scenes/matpreview.zip):
  scene geometry by Jonas Pilo, distributed by the Mitsuba project, which is
  maintained by Wenzel Jakob.

The Cornell, Utah, and Mitsuba source pages do not state a standard Creative
Commons license for these downloads; the manifest records `NOASSERTION`
instead of turning “public download” into a license claim. Their exact
converted OBJ files are included under the RenderFormer maintainers'
confirmed redistribution authority and are not relicensed by the repository
MIT license. Consult the source pages before reuse outside this release.

The primitives, backgrounds, lights, and logo copied unchanged from the
official Microsoft RenderFormer repository retain Microsoft's MIT notice in
[`licenses/microsoft-renderformer-MIT.txt`](../../licenses/microsoft-renderformer-MIT.txt).
This attribution document does not imply endorsement by any asset creator.
