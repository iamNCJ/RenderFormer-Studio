# RF2 paper-scene attribution and release status

The reference images in `reference/paper/` are RenderFormer-author outputs
copied from the project page. The scene configs and RenderFormer-authored
conversions are released under the repository license, subject to the
underlying asset terms below. Availability and required local inputs are listed
in [`README.md`](README.md).

## Bundled scene sources

- **Transparent Torus**: scene assembly and geometry by the RenderFormer
  authors. The scene was exported from Blender for RenderFormer.
- **Three Teapots**: adapted from *Veach, Ajar* by Benedikt Bitterli, released
  under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
  Geometry and materials were modified, simplified, and exported for
  RenderFormer. The exact paper input is `frame_000004` (17,653 tokens), whose
  third teapot has the rough SVBRDF shown in the paper. `frame_000003` has the
  same geometry and camera but a transmissive third teapot; the larger
  `frame_000001` is also not the paper input. The frame-4 additions include
  ambientCG [Asphalt 021](https://ambientcg.com/view?id=Asphalt021) on the
  third teapot and [Fabric 051](https://ambientcg.com/view?id=Fabric051) on
  the rear canvas. The table uses ambientCG
  [Wood Floor 060](https://ambientcg.com/view?id=WoodFloor060). All three are
  CC0; their maps were converted and resized by the RenderFormer exporter.
- **Displacement Mapped Cornell Cube**: Cornell short-box and tall-box geometry
  comes from the Cornell University Program of Computer Graphics
  [public-use data page](https://www.graphics.cornell.edu/online/box/data.html);
  Cornell does not state a standard open-content license for those source
  files. The RenderFormer maintainers approved redistribution of this derived,
  UV-mapped scene after crediting Cornell on the project website. The short-box
  material is TextureCan *Metal 0029* and the tall-box material is ambientCG
  *Planks 003*, both distributed through the pinned MatSynth source records
  under CC0 1.0. Background geometry is RenderFormer-authored (MIT), and the
  triangle light is derived from Microsoft RenderFormer (MIT).
- **Cube Pile**: procedural rigid-body cube geometry and scene assembly by the
  RenderFormer authors, generated in Blender and exported for RenderFormer.
- **Dragon**: *Dragon* by Delatronic, originally distributed as BlendSwap
  scene 80766 under
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The original
  BlendSwap URL no longer resolves; Benedikt Bitterli's maintained
  [Rendering Resources](https://benedikt-bitterli.me/resources/) page records
  the author and distributes the converted scene. The geometry was simplified
  and exported for RenderFormer.
- **Living Room**: *The Modern Living Room* by Wig42, originally distributed
  as BlendSwap scene 75692 under
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The original
  BlendSwap URL no longer resolves; Benedikt Bitterli's maintained
  [Rendering Resources](https://benedikt-bitterli.me/resources/) page records
  the author and distributes the converted scene. The scene was modified and
  exported for RenderFormer.
- **Living Room with Daylight** (bundled room assets only): *The White Room*
  by Jay-Artist, originally distributed as BlendSwap scene 41683 under
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The original
  BlendSwap URL no longer resolves; Benedikt Bitterli's maintained
  [Rendering Resources](https://benedikt-bitterli.me/resources/) page records
  the author and distributes the converted scene. The scene was modified and
  exported for RenderFormer. The paper HDRI is not covered by that scene
  license and is not bundled.
- **Bedroom** (bundled room assets only): *Modern Bedroom* by dylanheyes,
  from [Sketchfab](https://sketchfab.com/3d-models/modern-bedroom-ab1e4522dcae4af1bbd39299ab382f37),
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Geometry
  and materials were modified, selectively replaced, and exported for
  RenderFormer. Bundled replacement materials include ambientCG Fabric009,
  ambientCG Fabric076, and ShareTextures Marble38 through MatSynth, under
  CC0. The two paper wall base-color files have incomplete provenance and are
  not bundled.
- **Environment Lit Spheres** (bundled geometry/materials only): the floor
  uses TextureCan *Tiles 0068*, distributed through the pinned MatSynth
  dataset revision under CC0. The Grace Cathedral probe is copyrighted by
  Paul Debevec and is not bundled.

The `bedroom/frame/env_map.exr` file is a RenderFormer-generated constant
gray placeholder used by the recovered config with environment strength
zero; it is not a third-party HDRI.

The complete CC BY 3.0, CC BY 4.0, and CC0 license texts are included in
the repository's `licenses/` directory.

Original per-scene notices are kept beside the applicable scene directories.

## Inputs that are not redistributed

The following inputs are referenced by the scene configs but are not bundled:

- Grace Cathedral environment probe for **Environment Lit Spheres**.
- The paper-era `fluid_data_0001_preprocessed.vdb` for **Smoky Bunny**.
- Spaceship geometry and the historical fluid VDB for **Spaceship in Smoke**.
- The complete later candidate prepared frame for **Dinner Scene**. An older
  nine-mesh geometry lineage matches the paper token count exactly, but it is
  also withheld because its source authorship/license was not recorded and it
  does not contain the final textured material setup.
- The two bedroom wall base-color files.
- The daylight-room HDRI whose embedded filename is
  `20060807_wells6_hd.hdr`.

These files are deliberately absent because a public redistribution license
was not recovered. Supplying a matching local file to a launcher does not
change its copyright status. The later Dinner candidate has 29,992 geometry
triangles. The historical nine-mesh lineage has 29,275 faces plus one light
triangle, exactly matching the paper's 29,276 tokens, but token-count identity
does not establish asset rights or recover the final textured frame. Neither
is documented as a complete reproduction lock.

## Reference images

The gallery JPEGs document which scenes appeared in the paper and project
page. They are visual references, not model inputs or outputs generated by the
public launchers.
