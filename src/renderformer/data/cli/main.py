import argparse
import json
from pathlib import Path
import sys

from renderformer.data import __version__
from renderformer.data.capabilities.validator import validate_scene_for_export
from renderformer.data.h5.validate import validate_h5_file
from renderformer.data.schemas.io import load_export_profile, load_scene_semantic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="renderformer data",
        description="RenderFormer data generation and export tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"renderformer data {__version__}",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    generate = subcommands.add_parser("generate", help="Generate scenes and export H5 files.")
    generate_source = generate.add_mutually_exclusive_group(required=True)
    generate_source.add_argument("--scene-config")
    generate_source.add_argument("--template")
    generate_source.add_argument(
        "--prepared-frame",
        help="Blender-exported frame directory containing scene_config.json and split/",
    )
    generate.add_argument("--template-root")
    generate.add_argument("--object-list", help="Runtime object-list override for --template")
    generate.add_argument("--object-root", help="Runtime object-root override for --template")
    generate.add_argument("--texture-list", help="Runtime texture-list override for --template")
    generate.add_argument("--texture-root", help="Runtime texture-root override for --template")
    generate.add_argument("--env-map-list", help="Runtime environment-map list override for --template")
    generate.add_argument("--env-map-root", help="Runtime environment-map root override for --template")
    generate.add_argument(
        "--background-topology-debias",
        action="store_true",
        help=(
            "Apply the consolidated V1 background-topology correction policy; "
            "valid only with --template"
        ),
    )
    generate.add_argument(
        "--num-views",
        type=int,
        default=None,
        help=(
            "Number of cameras sampled by --template (default: 4); prepared frames "
            "always use the cameras stored in scene_config.json"
        ),
    )
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--profile", action="append", required=True)
    generate.add_argument("--resolution", type=int, default=256)
    generate.add_argument("--spp", type=int, default=4096)
    generate.add_argument("--texture-crop-res", type=int, default=192)
    generate.add_argument("--texture-target-res", type=int, default=128)
    generate.add_argument("--texture-size", type=int, default=64)
    generate.add_argument("--min-views", type=int, default=0)
    generate.add_argument("--test-mode", action="store_true")
    generate.add_argument("--no-heightmap-format", action="store_true")
    generate.add_argument("--save-images", action="store_true")
    generate.add_argument("--save-blend", action="store_true")
    generate.add_argument("--include-background", action="store_true")
    generate.add_argument("--skip-mesh-generation", action="store_true")

    generate_batch = subcommands.add_parser(
        "generate-batch",
        help="Plan or run a strict configs/data recipe with deterministic tasks.",
    )
    generate_batch.add_argument("--recipe", required=True)
    batch_action = generate_batch.add_mutually_exclusive_group()
    batch_action.add_argument("--describe", action="store_true")
    batch_action.add_argument("--validate", action="store_true")
    batch_action.add_argument(
        "--artifact-path",
        metavar="NAME",
        help=(
            "Print one historical training-artifact path and exit; this is a receipt "
            "resolver and does not materialize that artifact."
        ),
    )
    generate_batch.add_argument(
        "--output-dir",
        help=(
            "Directory for a new or resumed seeded primitive generation. Resume is "
            "accepted only when task plan, inputs, and outputs match. Its paths.txt "
            "is not one of the historical artifact/merged lists in the recipe."
        ),
    )
    generate_batch.add_argument("--asset-manifest")
    generate_batch.add_argument("--num-samples", type=int)
    generate_batch.add_argument("--seed", type=int)
    generate_batch.add_argument("--num-workers", type=int, default=1)
    generate_batch.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the deterministic task manifest; do not import Blender.",
    )
    generate_batch.add_argument(
        "--skip-asset-entry-check",
        action="store_true",
        help="Validate asset-list locations without checking every listed file.",
    )

    compose_rf1 = subcommands.add_parser(
        "compose-rf1",
        help="Compose a public legacy RF1 scene JSON into an H5 file without Blender.",
    )
    compose_rf1.add_argument("scene", help="Legacy RF1 scene JSON path.")
    compose_rf1.add_argument("--output", required=True, help="Output RF1 H5 path.")
    compose_rf1.add_argument(
        "--no-legacy-texture-quantization",
        action="store_true",
        help="Skip the historical float16 value quantization before writing float32.",
    )

    validate_scene = subcommands.add_parser("validate-scene", help="Validate a scene against an export profile.")
    validate_scene.add_argument("--scene", required=True)
    validate_scene.add_argument("--profile", required=True)

    validate_h5 = subcommands.add_parser("validate-h5", help="Validate an H5 file against a schema.")
    validate_h5.add_argument("--input", required=True)
    validate_h5.add_argument("--format", required=True, choices=["v1", "v2"])

    validate_assets = subcommands.add_parser(
        "validate-assets",
        help="Validate fixed and external assets referenced by scene templates.",
    )
    validate_assets.add_argument(
        "--template",
        action="append",
        default=[],
        help="Template JSONC path (repeatable).",
    )
    validate_assets.add_argument(
        "--template-root",
        help="Root for relative paths in all selected templates (default: infer per template).",
    )
    validate_assets.add_argument(
        "--training-manifest",
        action="append",
        default=[],
        help="Packaged training-template manifest name or absolute path (repeatable).",
    )
    validate_assets.add_argument(
        "--asset-manifest",
        help="JSON mapping packaged external/... placeholders to local list/root paths.",
    )
    validate_assets.add_argument(
        "--skip-entry-check",
        action="store_true",
        help="Check list/root availability without stat'ing every listed asset.",
    )
    validate_assets.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum number of validation issues to print (default: 50).",
    )

    shuffle_envmaps = subcommands.add_parser(
        "shuffle-envmaps",
        help="Generate the six deterministic RGB permutations used by V2 HDRI data.",
    )
    shuffle_envmaps.add_argument(
        "--input-list",
        required=True,
        help="Newline-delimited relative EXR stems; entries omit the .exr suffix.",
    )
    shuffle_envmaps.add_argument("--input-root", required=True)
    shuffle_envmaps.add_argument("--output-root", required=True)
    shuffle_envmaps.add_argument(
        "--output-list-name",
        default="envmaps-filtered-channel-shuffle.txt",
        help="List filename inside the atomically published output root.",
    )
    shuffle_envmaps.add_argument(
        "--manifest-name",
        default="channel-shuffle-manifest.json",
        help="Receipt filename inside the atomically published output root.",
    )
    shuffle_envmaps.add_argument(
        "--num-workers",
        type=int,
        help="Process count (default: min(CPU count, 32, source count)).",
    )

    prepare_objaverse = subcommands.add_parser(
        "prepare-objaverse",
        help=(
            "Build watertight, QSlim, UV-unwrapped Objaverse tiers and their "
            "metadata-derived object lists."
        ),
    )
    prepare_objaverse.add_argument(
        "--input-manifest",
        required=True,
        help=(
            "JSONL rows with id, local source path, and optional special=true."
        ),
    )

    prepare_objaverse.add_argument("--output-dir", required=True)
    prepare_objaverse.add_argument("--num-workers", type=int, default=1)
    prepare_objaverse.add_argument(
        "--watertight-backend",
        choices=("voxel", "cuda-sdf"),
        default="voxel",
        help="Portable voxel reconstruction is the public default.",
    )
    prepare_objaverse.add_argument("--voxel-resolution", type=int, default=256)
    prepare_objaverse.add_argument("--normalize-radius", type=float, default=0.45)
    prepare_objaverse.add_argument("--sdf-resolution", type=int, default=256)
    prepare_objaverse.add_argument("--sdf-scale", type=float, default=0.8)
    prepare_objaverse.add_argument("--device", default="cuda")
    prepare_objaverse.add_argument(
        "--allow-empty-special",
        action="store_true",
        help=(
            "Permit an empty objects-special.txt only when the refraction stage "
            "will not be generated."
        ),
    )

    material_spheres = subcommands.add_parser(
        "material-spheres",
        help="Plan, render, or finalize the sharded material-autoencoder EXR dataset.",
    )
    material_spheres.add_argument("--recipe", required=True)
    material_action = material_spheres.add_mutually_exclusive_group(required=True)
    material_action.add_argument(
        "--validate",
        action="store_true",
        help="Validate and describe the YAML without importing bpy.",
    )
    material_action.add_argument(
        "--plan",
        action="store_true",
        help="Write or verify the deterministic plan and tasks.jsonl without importing bpy.",
    )
    material_action.add_argument(
        "--render-shard",
        type=int,
        metavar="INDEX",
        help="Render one explicit zero-based shard in the active PyPI bpy environment.",
    )
    material_action.add_argument(
        "--finalize",
        action="store_true",
        help="Verify every planned EXR and publish material-training.jsonl.",
    )
    material_spheres.add_argument("--output-dir")
    material_spheres.add_argument(
        "--environment-map",
        help="Explicit lat-long EXR used for every material sphere.",
    )
    material_spheres.add_argument("--seed", type=int)
    material_spheres.add_argument("--num-shards", type=int)
    material_spheres.add_argument(
        "--device",
        choices=("cpu", "cuda", "optix", "hip", "metal", "oneapi"),
        default="cpu",
        help="Explicit Cycles backend; no GPU is auto-selected.",
    )
    material_spheres.add_argument(
        "--device-index",
        type=int,
        help="Required zero-based backend device index for non-CPU rendering.",
    )

    convert = subcommands.add_parser("convert", help="Convert or postprocess existing H5 files.")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument(
        "--texture-encoder",
        required=True,
        choices=["qwen_vae", "dc_ae", "raw"],
        help="Explicit encoder backend.",
    )
    convert.add_argument("--model-id")
    convert.add_argument(
        "--revision",
        help=(
            "Optional texture-encoder Hugging Face branch, tag, or commit; "
            "omit it to use the repository default"
        ),
    )
    convert.add_argument("--cache-dir")
    convert.add_argument("--local-files-only", action="store_true")
    convert.add_argument("--device")
    convert.add_argument("--torch-dtype", default="float32")
    convert.add_argument("--triangle-batch-size", type=int, default=256)
    convert.add_argument("--keep-raw", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        template_overrides = (
            args.object_list,
            args.object_root,
            args.texture_list,
            args.texture_root,
            args.env_map_list,
            args.env_map_root,
        )
        if args.template is None and any(value is not None for value in template_overrides):
            parser.error("template asset overrides require --template")
        if args.template is None and args.background_topology_debias:
            parser.error("--background-topology-debias requires --template")
        if args.prepared_frame is not None and args.num_views is not None:
            parser.error(
                "--num-views cannot be used with --prepared-frame; edit or select "
                "the cameras in the prepared scene_config.json"
            )
        from renderformer.data.pipelines.generate import run_blender_generate

        results = run_blender_generate(
            scene_config_path=args.scene_config,
            prepared_frame_path=args.prepared_frame,
            output_dir=args.output_dir,
            profile_paths=args.profile,
            template_path=args.template,
            template_root=args.template_root,
            object_list_path=args.object_list,
            object_root=args.object_root,
            texture_list_path=args.texture_list,
            texture_root=args.texture_root,
            env_map_list_path=args.env_map_list,
            env_map_root=args.env_map_root,
            background_topology_debias=args.background_topology_debias,
            num_views=args.num_views if args.num_views is not None else 4,
            resolution=args.resolution,
            spp=args.spp,
            texture_crop_res=args.texture_crop_res,
            texture_target_res=args.texture_target_res,
            texture_size=args.texture_size,
            min_views=args.min_views,
            test_mode=args.test_mode,
            use_heightmap_format=not args.no_heightmap_format,
            save_images=args.save_images,
            save_blend=args.save_blend,
            include_background=args.include_background,
            skip_mesh_generation=args.skip_mesh_generation,
        )
        for result in results:
            print(f"generated: {result.output_path}")
        return 0
    if args.command == "generate-batch":
        from renderformer.data.data_recipe import (
            describe_data_recipe,
            load_data_recipe,
            run_data_recipe_batch,
        )

        recipe = load_data_recipe(args.recipe)
        if args.describe:
            print(describe_data_recipe(recipe))
            return 0
        if args.artifact_path is not None:
            try:
                print(recipe.artifact(args.artifact_path).path)
            except KeyError as exc:
                parser.error(str(exc))
            return 0
        if args.validate:
            if args.asset_manifest is not None:
                # A one-sample dry plan performs the same metadata-driven asset
                # validation as a real run, without importing Blender or writing
                # into the requested dataset directory.
                from renderformer.data.assets import validate_template_assets

                report = validate_template_assets(
                    [recipe.template_path(template) for template in recipe.templates],
                    asset_manifest_path=args.asset_manifest,
                    check_entries=not args.skip_asset_entry_check,
                )
                if not report.ok:
                    for issue in report.issues:
                        print(f"- {issue}", file=sys.stderr)
                    return 1
            print(f"valid: {recipe.recipe_id}")
            return 0
        missing_runtime = [
            name
            for name, value in (
                ("--output-dir", args.output_dir),
                ("--num-samples", args.num_samples),
                ("--seed", args.seed),
            )
            if value is None
        ]
        if missing_runtime:
            parser.error(
                "generate-batch runtime requires " + ", ".join(missing_runtime)
            )
        result_path = run_data_recipe_batch(
            recipe,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            num_workers=args.num_workers,
            asset_manifest_path=args.asset_manifest,
            dry_run=args.dry_run,
            check_asset_entries=not args.skip_asset_entry_check,
        )
        print(f"generated: {result_path}")
        return 0
    if args.command == "compose-rf1":
        from renderformer.data.pipelines.compose_rf1 import compose_rf1_h5

        result = compose_rf1_h5(
            args.scene,
            args.output,
            legacy_texture_quantization=not args.no_legacy_texture_quantization,
        )
        print(f"generated: {result.output_path}")
        return 0
    if args.command == "validate-scene":
        scene = load_scene_semantic(args.scene)
        profile = load_export_profile(args.profile)
        issues = validate_scene_for_export(scene, profile)
        if issues:
            for issue in issues:
                print(issue.reason)
        else:
            print("valid")
        return 0
    if args.command == "validate-h5":
        validate_h5_file(args.input, args.format)
        print(f"valid: {args.input}")
        return 0
    if args.command == "validate-assets":
        from renderformer.data.assets import (
            FINAL_TRAINING_MANIFESTS,
            validate_template_assets,
        )
        from renderformer.data.blender.templates import iter_training_template_paths

        if args.max_errors < 1:
            parser.error("--max-errors must be at least 1")
        template_paths = [Path(path) for path in args.template]
        training_manifests = list(args.training_manifest)
        if not template_paths and not training_manifests:
            training_manifests = list(FINAL_TRAINING_MANIFESTS)
        for manifest_name in training_manifests:
            template_paths.extend(iter_training_template_paths(manifest_name))
        report = validate_template_assets(
            template_paths,
            asset_manifest_path=args.asset_manifest,
            template_root=args.template_root,
            check_entries=not args.skip_entry_check,
        )
        print(
            f"checked {report.template_count} template(s), "
            f"{report.collection_count} asset collection(s), "
            f"{report.checked_entry_count} list entries"
        )
        if report.ok:
            print("valid")
            return 0
        for issue in report.issues[: args.max_errors]:
            print(f"- {issue}")
        omitted = len(report.issues) - args.max_errors
        if omitted > 0:
            print(f"... {omitted} additional issue(s) omitted")
        print(f"asset validation failed: {len(report.issues)} issue(s)")
        return 1
    if args.command == "shuffle-envmaps":
        from renderformer.data.envmap_assets import (
            generate_channel_shuffled_env_maps,
        )

        result = generate_channel_shuffled_env_maps(
            input_list=args.input_list,
            input_root=args.input_root,
            output_root=args.output_root,
            output_list_name=args.output_list_name,
            manifest_name=args.manifest_name,
            num_workers=args.num_workers,
        )
        print(
            f"generated {result.output_count} environment maps from "
            f"{result.source_count} sources"
        )
        print(f"list: {result.output_list}")
        print(f"manifest: {result.manifest}")
        return 0
    if args.command == "prepare-objaverse":
        from renderformer.data.objaverse import prepare_objaverse_collection

        receipt = prepare_objaverse_collection(
            input_manifest=args.input_manifest,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            watertight_backend=args.watertight_backend,
            voxel_resolution=args.voxel_resolution,
            normalize_radius=args.normalize_radius,
            sdf_resolution=args.sdf_resolution,
            sdf_scale=args.sdf_scale,
            device=args.device,
            allow_empty_special=args.allow_empty_special,
        )
        print(f"receipt: {receipt}")
        return 0
    if args.command == "material-spheres":
        from renderformer.data.material_spheres import (
            describe_material_sphere_recipe,
            finalize_material_spheres,
            plan_material_spheres,
            render_material_sphere_shard,
        )

        if args.validate:
            print(json.dumps(describe_material_sphere_recipe(args.recipe), indent=2))
            return 0
        missing = [
            name
            for name, value in (
                ("--output-dir", args.output_dir),
                ("--environment-map", args.environment_map),
                ("--seed", args.seed),
                ("--num-shards", args.num_shards),
            )
            if value is None
        ]
        if missing:
            parser.error("material-spheres runtime requires " + ", ".join(missing))
        kwargs = {
            "recipe_path": args.recipe,
            "environment_map": args.environment_map,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "num_shards": args.num_shards,
        }
        if args.plan:
            result = plan_material_spheres(**kwargs)
            print(f"plan: {result}")
            return 0
        if args.render_shard is not None:
            result = render_material_sphere_shard(
                **kwargs,
                shard_index=args.render_shard,
                device=args.device,
                device_index=args.device_index,
            )
            print(f"shard receipt: {result}")
            return 0
        result = finalize_material_spheres(**kwargs)
        print(f"training manifest: {result}")
        return 0
    if args.command == "convert":
        from renderformer.data.textures.postprocess import postprocess_v2_h5

        encoder_config = {
            "type": args.texture_encoder,
            "triangle_batch_size": args.triangle_batch_size,
            "torch_dtype": args.torch_dtype,
            "local_files_only": args.local_files_only,
        }
        if args.model_id is not None:
            encoder_config["model_id"] = args.model_id
        if args.revision is not None:
            encoder_config["revision"] = args.revision
        if args.cache_dir is not None:
            encoder_config["cache_dir"] = args.cache_dir
        if args.device is not None:
            encoder_config["device"] = args.device
        postprocess_v2_h5(Path(args.input), Path(args.output), encoder_config, keep_raw=args.keep_raw)
        print(f"converted: {args.output}")
        return 0
if __name__ == "__main__":
    raise SystemExit(main())
