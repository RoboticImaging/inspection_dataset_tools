"""Run VGGT-SLAM on one sequence of an inspection asset and write depth + camera predictions back
into the asset folder (predictions/{method}/...), in the same format the ground truth is stored
in -- so inspection_viz's Source toggle picks them up unchanged.

Unlike baselines/vggt/run_sequence.py's single joint inference call, this drives VGGT-SLAM's own
submap/loop-closure pipeline (vggt_slam.solver.Solver): frames are grouped into overlapping
submaps, each run through VGGT, then stitched and loop-closed by the SLAM backend. Same
predictions/{method}/{depths,cam_params,clouds}/... convention as the other baselines, via
baselines.common.runner.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/old_house.yaml --model ../configs/vggt_slam.yaml
    python run_sequence.py ../configs/vggt-slam/vggt-slam-example-config.yaml  # both halves in one file

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/vggt-slam install.
"""

import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver

from vggt.models.vggt import VGGT as VGGTModel  # aliased: the Baseline below owns the name

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from inspection_viz.dataset import Frame

# vggt_slam.solver.Solver.run_predictions calls load_and_preprocess_images() with no `mode` arg,
# i.e. the default "crop" mode, which always resizes to a fixed 518px width -- that's the frame
# width the solved intrinsics (K) below are measured against.
VGGT_IMAGE_WIDTH = 518


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
class ModelConfig:
    checkpoint_url: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    run: RunConfig = dataclasses.field(default_factory=RunConfig)
    vis: VisConfig = dataclasses.field(default_factory=VisConfig)


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


def write_all_frames(out: SequenceOutput, solver, depths_by_id: dict[int, np.ndarray],
                     orig_image_size: tuple[int, int], vggt_image_size: tuple[int, int],
                     align: np.ndarray) -> None:
    """Writes depth+camera together, one call per solved frame -- must wait until `align` is
    known (only available after every submap is solved and compute_gt_alignment can run), so
    depths collected progressively during the submap loop (see collect_depths) are buffered in
    `depths_by_id` until here instead of written immediately.

    Reuses the exact same per-frame projection-matrix decomposition
    vggt_slam.map.GraphMap.write_poses_to_file does internally (same submap iteration, same
    decompose_camera call) -- just keeps K (intrinsics) here instead of discarding it, and skips
    only via the public GraphMap/Submap API so nothing under vggt_slam/ needs editing.
    """
    graph = solver.graph
    graph_map = solver.map
    all_poses = graph_map.get_all_cam_matricies(graph=graph, give_camera_mat=True)
    width, height = orig_image_size

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
            c2w = align @ c2w  # lands every pose in the same world frame gt is in

            frame_id = int(frame_id)
            depth = cv2.resize(depths_by_id[frame_id], (width, height), interpolation=cv2.INTER_LINEAR)
            out.write_frame(frame_id, depth, c2w, K, intrinsics_width=width)


def collect_depths(depths_by_id: dict[int, np.ndarray], predictions: dict, frame_ids: list[int]) -> None:
    """Buffers one submap's raw (VGGT processing-resolution) depth per frame id -- overlapping
    frame ids (shared between consecutive submaps) take the later submap's prediction, same as
    write_all_frames's own poses/intrinsics resolve the same overlap."""
    assert len(predictions["depth"]) == len(frame_ids)
    for depth, frame_id in zip(predictions["depth"], frame_ids):
        depths_by_id[frame_id] = depth.squeeze().astype(np.float32)


def save_points(out: SequenceOutput, solver, align: np.ndarray) -> None:
    """Writes the fused world-space point cloud, in the same gt-aligned world frame
    write_all_frames writes the poses in.

    Reimplements vggt_slam.map.GraphMap.write_points_to_file (same submap iteration, same
    get_points_in_world_frame/get_points_colors calls) only via the public GraphMap/Submap API --
    same rationale, and same "nothing under vggt_slam/ needs editing" constraint, as
    write_all_frames -- but applies `align` to every point before writing, and writes through
    `out.write_cloud` (matches predictions/da3/clouds/... on disk) instead of writing straight to
    open3d.
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
    out.write_cloud(points_all, colors_all.astype(np.uint8))


class VGGTSLAM(Baseline):
    """VGGT-SLAM, driven through its own submap/loop-closure pipeline (see module docstring)."""

    name = "vggt-slam"
    Config = ModelConfig

    def setup(self) -> None:
        cfg = self.config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Initializing and loading VGGT model...")
        self.model = VGGTModel()
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(cfg.checkpoint_url))
        self.model.eval()
        self.model = self.model.to(torch.bfloat16)  # use half precision
        self.model = self.model.to(self.device)

        # Semantic-query models, loaded once and reused across sequences same as self.model --
        # but note run_os's own interactive REPL (see run(), below) blocks until you quit it, so
        # a data config selecting several sequences won't move on to the next one until then.
        if cfg.run.run_os:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms

            sam3_model = build_sam3_image_model()
            self.processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

            self.clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # downloads from HF
            self.clip_model = self.clip_model.cuda()
            self.clip_tokenizer = transforms.get_text_tokenizer(self.clip_model.context_length)
            self.clip_preprocess = transforms.get_image_transform(self.clip_model.image_size)
        else:
            self.clip_model = self.clip_preprocess = self.clip_tokenizer = None

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        cfg = self.config
        frames_by_id = {f.index: f for f in frames}

        solver = Solver(
            init_conf_threshold=cfg.run.conf_threshold,
            lc_thres=cfg.run.lc_thres,
            vis_voxel_size=cfg.vis.vis_voxel_size,
            vis_imgs=cfg.vis.vis_imgs,
        )

        use_optical_flow_downsample = True
        print(f"Running {len(frames)} frames ...")

        frame_subset: list[Frame] = []
        depths_by_id: dict[int, np.ndarray] = {}
        submap_count = 0
        frame_count = 0
        total_time_start = time.time()
        keyframe_time = utils.Accumulator()
        backend_time = utils.Accumulator()

        # (width, height) pairs, captured once each and reused for every frame: the source video
        # is fixed-resolution, and VGGT batches all frames of a submap through the same
        # preprocessing.
        orig_image_size = None
        vggt_image_size = None

        for frame in tqdm(frames):
            if use_optical_flow_downsample:
                with keyframe_time:
                    img = cv2.imread(str(frame.rgb_path))
                    if orig_image_size is None:
                        orig_image_size = (img.shape[1], img.shape[0])
                    enough_disparity = solver.flow_tracker.compute_disparity(
                        img, cfg.run.min_disparity, cfg.vis.vis_flow)
                    if enough_disparity:
                        frame_subset.append(frame)
                        frame_count += 1
            else:
                frame_subset.append(frame)

            # Run submap processing if enough images are collected or it's the last group.
            if len(frame_subset) == cfg.run.submap_size + cfg.run.overlapping_window_size \
                    or frame is frames[-1]:
                submap_count += 1
                image_names_subset = [str(f.rgb_path) for f in frame_subset]
                t1 = time.time()
                predictions = solver.run_predictions(
                    image_names_subset, self.model, cfg.run.max_loops,
                    self.clip_model, self.clip_preprocess)

                if vggt_image_size is None:
                    # predictions['images'] is (S, 3, H, W) -- see Solver.run_predictions.
                    vggt_image_size = (predictions["images"].shape[-1], predictions["images"].shape[-2])

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

                collect_depths(depths_by_id, predictions, [f.index for f in frame_subset])

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
            solver.viewer.add_object_query_gui(solver, self.clip_model, self.clip_tokenizer,
                                               self.processor, data_lock)

            while True:
                query = input("\nEnter text query or q to quit: ").strip()
                if len(query) == 0:
                    print("Empty query. Exiting.")
                    return
                if query == "q":
                    print("Exiting.")
                    break

                text_emb = utils.compute_text_embeddings(self.clip_model, self.clip_tokenizer, query)
                overall_best_score, overall_best_submap_id, overall_best_frame_index = \
                    solver.map.retrieve_best_semantic_frame(text_emb)

                found_submap = solver.map.get_submap(overall_best_submap_id)

                best_img = found_submap.get_frame_at_index(overall_best_frame_index)
                print("Score:", overall_best_score)
                with torch.no_grad():
                    best_img = to_pil_image(best_img)
                    inference_state = self.processor.set_image(best_img)
                    output = self.processor.set_text_prompt(state=inference_state, prompt=query)
                    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                    print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                    print("Scores:", scores.cpu().numpy())

                masked_img = utils.overlay_masks(best_img, masks)
                masked_img.show()

                for i in range(masks.shape[0]):
                    mask = masks[i].cpu().numpy()
                    obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(
                        found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                    solver.viewer.visualize_obb(
                        center=obb_center, extent=obb_extent, rotation=obb_rotation,
                        color=(255, 0, 0), line_width=8.0,
                    )

        if not cfg.vis.vis_map:
            # just show the map after all submaps have been processed
            solver.update_all_submap_vis()

        align = compute_gt_alignment(solver, frames_by_id)
        write_all_frames(out, solver, depths_by_id, orig_image_size, vggt_image_size, align)
        if out.spec.cloud:
            save_points(out, solver, align)
            print(f"  wrote cloud -> {out.cloud_path.relative_to(out.root)}")


if __name__ == "__main__":
    cli_main(VGGTSLAM, description=__doc__)
