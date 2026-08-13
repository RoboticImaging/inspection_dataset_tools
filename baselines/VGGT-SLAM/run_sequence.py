import os
import glob
import time
import argparse
import cv2
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap

from vggt.models.vggt import VGGT
import yaml

from baselines.common.config import DatasetSelection
from baselines.common.writer import frame_stem, write_cam, write_cloud
from inspection_viz.dataset import AssetDataset

# vggt_slam.solver.Solver.run_predictions calls load_and_preprocess_images() with no `mode` arg,
# i.e. the default "crop" mode, which always resizes to a fixed 518px width -- that's the frame
# width the solved intrinsics (K) below are measured against.
VGGT_IMAGE_WIDTH = 518


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--config", type=str, required=True, help="Path to config file")

def save_depths(log_path, predictions, image_names_subset, orig_image_size):

    assert len(predictions['depth']) == len(image_names_subset)
    assert len(predictions['depth_conf']) == len(image_names_subset)

    width, height = orig_image_size
    for depth, depth_conf, im_name in zip(predictions['depth'], predictions['depth_conf'], image_names_subset):
        stem = os.path.splitext(os.path.basename(im_name))[0]
        depth = depth.squeeze().astype(np.float32)
        depth_conf = depth_conf.squeeze().astype(np.float32)

        # VGGT predicts at its own fixed processing resolution (see VGGT_IMAGE_WIDTH), not the
        # original frame size -- resize back before writing, same rationale as the intrinsics
        # rescale in save_cam_params.
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        depth_conf = cv2.resize(depth_conf, (width, height), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(log_path, f"{stem}.exr"), depth)
        cv2.imwrite(os.path.join(log_path, f"{stem}_conf.exr"), depth_conf)

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


def get_gt_c2w(cfg, frame_id):
    """Ground-truth camera-to-world pose for `frame_id`, as a (4,4) OpenCV matrix.

    Uses the same `DatasetSelection`/`AssetDataset` infra `depth_anything_3/run_sequence.py`
    uses, so this baseline's gt lookup follows the exact same dataset-layout convention.
    """
    if 'soil' in cfg['dataset']:
        assert 'ref' not in cfg['dataset'], "cfg should only contain one of soil or ref"
        soil, reflectivity = cfg['dataset']['soil'], None
    elif 'ref' in cfg['dataset']:
        soil, reflectivity = None, cfg['dataset']['ref']
    else:
        raise KeyError("Either soil or ref needed in config file")

    variant = DatasetSelection(
        root=cfg['dataset']['root'],
        trajectory=cfg['dataset']['trajectory'],
        illumination=cfg['dataset']['illumination'],
        soil=soil,
        reflectivity=reflectivity,
    ).variant()
    seq = AssetDataset(cfg['dataset']['root']).sequence(variant, source="gt")
    frame = next((f for f in seq if f.index == frame_id), None)
    if frame is None or frame.camera() is None:
        raise KeyError(f"no gt camera for frame {frame_id} in {cfg['dataset']['root']}")
    return frame.camera().c2w


def compute_gt_alignment(solver, cfg):
    """The (4,4) rigid transform that puts VGGT-SLAM's world gauge onto the gt one, as far as one
    frame can. Like DA3 (see depth_anything_3/run_sequence.py's `_gt_frame0_alignment`),
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
        gt_c2w0 = get_gt_c2w(cfg, int(frame_ids[0]))
        return gt_c2w0 @ np.linalg.inv(pred_c2w0)

    raise RuntimeError("no non-loop-closure submap with frames to align against gt")


def save_cam_params(log_path_cam_params, solver, orig_image_size, vggt_image_size, align):
    """Write one cam_params JSON per solved frame, in the same loc/rot/focal_length/sensor_width
    schema `baselines/common/writer.write_cam` writes for the other baselines (matches
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

            write_cam(
                Path(log_path_cam_params) / f"{frame_stem(int(frame_id))}.json",
                c2w, K, width=orig_image_size[0],
            )


def save_points(log_path_clouds, solver, align):
    """Write the fused world-space point cloud, in the same gt-aligned world frame `save_cam_params`
    writes the poses in.

    Reimplements vggt_slam.map.GraphMap.write_points_to_file (same submap iteration, same
    get_points_in_world_frame/get_points_colors calls) only via the public GraphMap/Submap API --
    same rationale, and same "nothing under vggt_slam/ needs editing" constraint, as
    save_cam_params -- but applies `align` to every point before writing, and writes through the
    shared baselines.common.writer convention (matches predictions/da3/clouds/... on disk) instead
    of writing straight to open3d.
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
    write_cloud(Path(log_path_clouds), points_all, colors_all.astype(np.uint8))


def get_image_and_save_folders(cfg):
    root = cfg['dataset']['root']
    illumination = cfg['dataset']['illumination']

    # Either soil or reflectivity (ref) will be in the dataset
    # Here we handle which is present and carry this forward to forming paths
    if 'soil' in cfg['dataset']:
        assert 'ref' not in cfg['dataset'], "cfg should only contain one of soil or ref"
        var_key = 'soil'
        var_key_value = cfg['dataset']['soil']
    elif 'ref' in cfg['dataset']:
        var_key = 'ref'
        var_key_value = cfg['dataset']['ref']
    else:
        raise KeyError("Either soil or ref needed in config file")

    image_folder = os.path.join(
            cfg['dataset']['root'], cfg['dataset']['illumination'],
            f"{var_key}={var_key_value},traj={cfg['dataset']['trajectory']}",
        )

    log_path_cam_params = os.path.join(
            cfg['dataset']['root'],
            "predictions", cfg['output']['method'],
            cfg['dataset']['illumination'],
            "cam_params",
            f"{var_key}={var_key_value},traj={cfg['dataset']['trajectory']}",
        )
    log_path_depth = os.path.join(
                cfg['dataset']['root'],
                "predictions", cfg['output']['method'],
                cfg['dataset']['illumination'],
                "depths",
                f"{var_key}={var_key_value},traj={cfg['dataset']['trajectory']}",
            )
    log_path_clouds = os.path.join(
                    cfg['dataset']['root'],
                    "predictions", cfg['output']['method'],
                    cfg['dataset']['illumination'],
                    "clouds",
                    f"{var_key}={var_key_value},traj={cfg['dataset']['trajectory']}.ply",
                )

    return image_folder, log_path_cam_params, log_path_depth, log_path_clouds

def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Derived from dataset/output config -- there's no literal image_folder/log_path/log_results
    # key in the config, so these are built from the asset layout convention instead.
    image_folder, log_path_cam_params, log_path_depth, log_path_clouds = get_image_and_save_folders(cfg)
    os.makedirs(log_path_cam_params, exist_ok=True)
    os.makedirs(log_path_depth, exist_ok=True)
    os.makedirs(os.path.dirname(log_path_clouds), exist_ok=True)

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=cfg['run']['conf_threshold'],
        lc_thres=cfg['run']['lc_thres'],
        vis_voxel_size=cfg['vis']['vis_voxel_size'],
        vis_imgs=cfg['vis']['vis_imgs']
    )


    print("Initializing and loading VGGT model...")


    if cfg['run']['run_os']:
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
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(image_folder, "*"))
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower()
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    used_image_names = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()

    # (width, height) pairs, captured once each and reused for every frame: the source video is
    # fixed-resolution, and VGGT batches all frames of a submap through the same preprocessing.
    orig_image_size = None
    vggt_image_size = None

    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                if orig_image_size is None:
                    orig_image_size = (img.shape[1], img.shape[0])
                enough_disparity = solver.flow_tracker.compute_disparity(img, cfg['run']['min_disparity'], cfg['vis']['vis_flow'])
                if enough_disparity:
                    image_names_subset.append(image_name)
                    used_image_names.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == cfg['run']['submap_size'] + cfg['run']['overlapping_window_size'] or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, cfg['run']['max_loops'], clip_model, clip_preprocess)

            if vggt_image_size is None:
                # predictions['images'] is (S, 3, H, W) -- see vggt_slam.solver.Solver.run_predictions.
                vggt_image_size = (predictions['images'].shape[-1], predictions['images'].shape[-2])

            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if cfg['vis']['vis_map']:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            save_depths(log_path_depth, predictions, image_names_subset, orig_image_size)

            # Reset for next submap.
            image_names_subset = image_names_subset[-cfg['run']['overlapping_window_size']:]




    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
    print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())


    if cfg['run']['run_os']:
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

    if not cfg['vis']['vis_map']:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    align = compute_gt_alignment(solver, cfg)
    save_cam_params(log_path_cam_params, solver, orig_image_size, vggt_image_size, align)
    save_points(log_path_clouds, solver, align)

    # if not skip_dense_log:
    #     # Log the full point cloud as one file
    #     solver.map.write_points_to_file(solver.graph, log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
