"""Run VGGT-SLAM on one sequence of an inspection asset and write depth + camera predictions back
into the asset folder (predictions/{method}/...), in the same format the ground truth is stored
in -- so inspection_viz's Source toggle picks them up unchanged.

Unlike baselines/vggt/run_sequence.py's single joint inference call, this drives VGGT-SLAM's own
submap/loop-closure pipeline (vggt_slam.solver.Solver): frames are grouped into overlapping
submaps, each run through VGGT, then stitched and loop-closed by the SLAM backend. Same
dataset/output config and predictions/{method}/{depths,cam_params,clouds}/... convention as the
other baselines, though -- see baselines.common.config/baselines.common.writer.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py ../configs/vggt-slam/vggt-slam-example-config.yaml

Paths are the same inside the container as on the host, so an absolute config path works too.
Do not run this outside the container: it needs the image's torch/vggt-slam install.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines.common

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver

from vggt.models.vggt import VGGT

from baselines.common.config import DatasetSelection, OutputSpec, load_config
from baselines.common.writer import (
    cloud_path, frame_stem, output_dirs, write_cam, write_cloud, write_depth,
)
from inspection_viz.dataset import AssetDataset, Frame

# vggt_slam.solver.Solver.run_predictions calls load_and_preprocess_images() with no `mode` arg,
# i.e. the default "crop" mode, which always resizes to a fixed 518px width -- that's the frame
# width the solved intrinsics (K) below are measured against.
VGGT_IMAGE_WIDTH = 518


@dataclasses.dataclass
class ModelConfig:
    checkpoint_url: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


@dataclasses.dataclass
class RunConfig:
    conf_threshold: float = 25.0
    lc_thres: float = 0.95
    min_disparity: float = 50
    submap_size: int = 16
    overlapping_window_size: int = 1
    max_loops: int = 1
    run_os: bool = False  # open-set semantic query mode (needs sam3/PE, interactive at the end)


@dataclasses.dataclass
class VisConfig:
    vis_map: bool = False
    vis_imgs: bool = False
    vis_voxel_size: float | None = None
    vis_flow: bool = False


@dataclasses.dataclass
class Config:
    dataset: DatasetSelection = dataclasses.field(default_factory=DatasetSelection)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)
    vis: VisConfig = dataclasses.field(default_factory=VisConfig)
    output: OutputSpec = dataclasses.field(default_factory=OutputSpec)


def save_depths(depth_dir: Path, predictions: dict, frame_ids: list[int], orig_image_size: tuple[int, int]) -> None:
    assert len(predictions['depth']) == len(frame_ids)

    width, height = orig_image_size
    for depth, frame_id in zip(predictions['depth'], frame_ids):
        depth = depth.squeeze().astype(np.float32)

        # VGGT predicts at its own fixed processing resolution (see VGGT_IMAGE_WIDTH), not the
        # original frame size -- resize back before writing, same rationale as the intrinsics
        # rescale in save_cam_params.
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        write_depth(depth_dir / f"{frame_stem(frame_id)}.exr", depth)


def rescale_vggt_intrinsics_from_K(K_pred, orig_image_size, vggt_image_size):
    """Rescales a VGGT 3x3 intrinsic matrix -- measured against the (width, height) VGGT actually
    processed -- back to the original frame's (width, height).

    VGGT's default "crop" preprocessing (see VGGT_IMAGE_WIDTH) uniformly scales width to 518px
    and only center-crops height when the source is taller than it is wide (see
    vggt.utils.load_fn.load_and_preprocess_images); for the landscape video this baseline
    targets, height stays under 518 and that crop never fires, so recovering a uniform scale
    from the *actual* processed shape (not the nominal 518 constant, which the divisible-by-14
    rounding perturbs slightly) is exact.
    """
    w_orig, h_orig = orig_image_size
    w_vggt, h_vggt = vggt_image_size
    scale_x, scale_y = w_orig / w_vggt, h_orig / h_vggt

    K_orig = K_pred.copy()
    K_orig[0, 0] *= scale_x  # fx
    K_orig[1, 2] *= scale_y  # cy
    K_orig[1, 1] *= scale_y  # fy
    K_orig[0, 2] *= scale_x  # cx
    return K_orig


def compute_gt_alignment(solver, frames_by_id: dict[int, Frame]) -> np.ndarray:
    """The (4,4) rigid transform that puts VGGT-SLAM's world gauge onto the gt one, as far as one
    frame can. Like the other baselines' `_gt_frame0_alignment` (see baselines/vggt/run_sequence.py),
    VGGT-SLAM reasons in its own arbitrary gauge -- world frame anchored to whichever frame it
    solved first, at an unrecoverable scale (the inter-submap `scale_factor` in vggt_slam/solver.py
    only reconciles submaps against *each other*, not against gt) -- so raw predictions land
    nowhere near the gt trajectory even though the trajectory's shape is right. This is the
    relative transform between the gt and predicted pose of the first solved frame; applying it
    lands that frame exactly on its gt pose and fixes every other frame's position/orientation
    too, but not VGGT-SLAM's own scale.
    """
    graph = solver.graph
    graph_map = solver.map
    all_poses = graph_map.get_all_cam_matricies(graph=graph, give_camera_mat=True)

    for submap in graph_map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        frame_ids = submap.get_frame_ids()
        if not frame_ids:
            continue
        _, rotation_matrix, t, _ = utils.decompose_camera(all_poses[0])
        pred_c2w0 = np.eye(4)
        pred_c2w0[:3, :3] = rotation_matrix
        pred_c2w0[:3, 3] = t
        gt_c2w0 = frames_by_id[int(frame_ids[0])].camera().c2w
        return gt_c2w0 @ np.linalg.inv(pred_c2w0)

    raise RuntimeError("no non-loop-closure submap with frames to align against gt")


def save_cam_params(cam_dir: Path, solver, orig_image_size, vggt_image_size, align):
    """Write one cam_params JSON per solved frame, in the same loc/rot/focal_length/sensor_width
    schema `baselines.common.writer.write_cam` writes for the other baselines (matches
    predictions/da3/cam_params/... on disk).

    Reuses the exact same per-frame projection-matrix decomposition
    vggt_slam.map.GraphMap.write_poses_to_file does internally (same submap iteration, same
    decompose_camera call) -- just keeps K (intrinsics) here instead of discarding it, and skips
    only via the public GraphMap/Submap API so nothing under vggt_slam/ needs editing.

    `align` (from `compute_gt_alignment`) is applied to every pose so the poses land in the same
    world frame as gt -- and thus the same world frame the other baselines' predictions do.
    """
    graph = solver.graph
    graph_map = solver.map
    all_poses = graph_map.get_all_cam_matricies(graph=graph, give_camera_mat=True)

    count = 0
    for submap in graph_map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        for frame_id in submap.get_frame_ids():
            K, rotation_matrix, t, scale = utils.decompose_camera(all_poses[count])
            K = rescale_vggt_intrinsics_from_K(K, orig_image_size, vggt_image_size)
            count += 1

            c2w = np.eye(4)
            c2w[:3, :3] = rotation_matrix
            c2w[:3, 3] = t
            c2w = align @ c2w

            write_cam(cam_dir / f"{frame_stem(int(frame_id))}.json", c2w, K, width=orig_image_size[0])


def save_points(ply_path: Path, solver, align):
    """Write the fused world-space point cloud, in the same gt-aligned world frame `save_cam_params`
    writes the poses in.

    Reimplements vggt_slam.map.GraphMap.write_points_to_file (same submap iteration, same
    get_points_in_world_frame/get_points_colors calls) only via the public GraphMap/Submap API --
    same rationale, and same "nothing under vggt_slam/ needs editing" constraint, as
    save_cam_params -- but applies `align` to every point before writing, and writes through
    `baselines.common.writer.write_cloud` (matches predictions/da3/clouds/... on disk) instead of
    writing straight to open3d.
    """
    graph = solver.graph
    graph_map = solver.map

    points_all, colors_all = [], []
    for submap in graph_map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        points = submap.get_points_in_world_frame(graph).reshape(-1, 3)
        points = points @ align[:3, :3].T + align[:3, 3]
        points_all.append(points)
        colors_all.append(submap.get_points_colors())

    points_all = np.concatenate(points_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0).astype(np.float32)
    # GraphMap.write_points_to_file normalizes the other way (>1.0 -> /255) for open3d, which
    # wants floats in [0, 1]; write_cloud wants uint8 in [0, 255], so undo that here instead.
    if colors_all.max() <= 1.0:
        colors_all = colors_all * 255.0
    write_cloud(ply_path, points_all, colors_all.astype(np.uint8))


def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="path to a run config yaml, e.g. ../configs/vggt-slam/vggt-slam-example-config.yaml")
    args = p.parse_args()
    cfg = load_config(args.config, Config)

    ds = AssetDataset(cfg.dataset.root)
    variant = cfg.dataset.variant()
    frames = cfg.dataset.select(ds.sequence(variant))  # gt sequence: rgb input
    if not frames:
        raise SystemExit(f"no frames selected for {variant} in {cfg.dataset.root}")
    frames_by_id = {f.index: f for f in frames}

    depth_dir, cam_dir = output_dirs(ds.root, ds.layout, cfg.output.method, variant.as_dict())
    ply_path = cloud_path(ds.root, ds.layout, cfg.output.method, variant.as_dict())
    if not cfg.output.overwrite and depth_dir.exists() and any(depth_dir.iterdir()):
        raise SystemExit(f"{depth_dir} already has output; set output.overwrite: true to replace")

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=cfg.run.conf_threshold,
        lc_thres=cfg.run.lc_thres,
        vis_voxel_size=cfg.vis.vis_voxel_size,
        vis_imgs=cfg.vis.vis_imgs,
    )

    print("Initializing and loading VGGT model...")

    if cfg.run.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(cfg.model.checkpoint_url))
    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)

    print(f"Running {len(frames)} frames from {cfg.dataset.root} ...")

    frame_subset: list[Frame] = []
    submap_count = 0
    frame_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()

    # (width, height) pairs, captured once each and reused for every frame: the source video is
    # fixed-resolution, and VGGT batches all frames of a submap through the same preprocessing.
    orig_image_size = None
    vggt_image_size = None

    for frame in tqdm(frames):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(str(frame.rgb_path))
                if orig_image_size is None:
                    orig_image_size = (img.shape[1], img.shape[0])
                enough_disparity = solver.flow_tracker.compute_disparity(img, cfg.run.min_disparity, cfg.vis.vis_flow)
                if enough_disparity:
                    frame_subset.append(frame)
                    frame_count += 1
        else:
            frame_subset.append(frame)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(frame_subset) == cfg.run.submap_size + cfg.run.overlapping_window_size or frame is frames[-1]:
            submap_count += 1
            image_names_subset = [str(f.rgb_path) for f in frame_subset]
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, cfg.run.max_loops, clip_model, clip_preprocess)

            if vggt_image_size is None:
                # predictions['images'] is (S, 3, H, W) -- see vggt_slam.solver.Solver.run_predictions.
                vggt_image_size = (predictions['images'].shape[-1], predictions['images'].shape[-2])

            print("Solver total time", time.time() - t1)
            print(submap_count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if cfg.vis.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            save_depths(depth_dir, predictions, [f.index for f in frame_subset], orig_image_size)

            # Reset for next submap.
            frame_subset = frame_subset[-cfg.run.overlapping_window_size:]

    total_time = time.time() - total_time_start
    average_fps = total_time / frame_count
    print(frame_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / frame_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / frame_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / frame_count)
    print("Average backend time per frame:", backend_time.total_time / frame_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / frame_count)
    print("Average total time per frame:", total_time / frame_count)
    print("Average FPS:", 1 / average_fps)

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if cfg.run.run_os:
        # Register the viser object-query panel so the user can search for
        # objects in the viewer in addition to the terminal prompt below.
        import threading
        data_lock = threading.Lock()
        solver.viewer.add_object_query_gui(solver, clip_model, clip_tokenizer, processor, data_lock)

        while True:
            # Prompt user for text input
            query = input("\nEnter text query or q to quit: ").strip()
            if len(query) == 0:
                print("Empty query. Exiting.")
                return

            if query == "q":
                print("Exiting.")
                return

            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            overall_best_score, overall_best_submap_id, overall_best_frame_index = solver.map.retrieve_best_semantic_frame(text_emb)

            found_submap = solver.map.get_submap(overall_best_submap_id)

            # Display image
            best_img = found_submap.get_frame_at_index(overall_best_frame_index)
            print("Score:", overall_best_score)
            with torch.no_grad():
                # convert torch image to PIL
                best_img = to_pil_image(best_img)
                inference_state = processor.set_image(best_img)
                output = processor.set_text_prompt(state=inference_state, prompt=query)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                print("Scores:", scores.cpu().numpy())

            masked_img = utils.overlay_masks(best_img, masks)
            masked_img.show()

            for i in range(masks.shape[0]):
                mask = masks[i].cpu().numpy()
                obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                solver.viewer.visualize_obb(
                    center=obb_center,
                    extent=obb_extent,
                    rotation=obb_rotation,
                    color=(255, 0, 0),
                    line_width=8.0,
                )

    if not cfg.vis.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    align = compute_gt_alignment(solver, frames_by_id)
    save_cam_params(cam_dir, solver, orig_image_size, vggt_image_size, align)
    if cfg.output.cloud:
        save_points(ply_path, solver, align)
        print(f"  wrote cloud -> {ply_path.relative_to(ds.root)}")

    print(f"done: predictions/{cfg.output.method}/ under {ds.root}")


if __name__ == "__main__":
    main()
