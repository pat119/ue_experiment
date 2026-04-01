import torch
import cv2
import numpy as np
import os, sys, time, re, threading
import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image
import unrealcv
import itertools
import json
import zlib
import base64
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# --- Configuration ---
OUTPUT_BASE = "/path/to/output/directory"
CHECKPOINT_PATH = "/path/to/sam_pth_file/"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing SAM on {DEVICE}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
print("SAM loaded successfully.")

CHECKPOINT_FILE = "experiment_checkpoint.json"
RESULTS_LOG = os.path.join(OUTPUT_BASE, "experiment_results.jsonl")

# --- Checkpointing Logic ---
def save_checkpoint(t_idx, iter_idx):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"t_idx": t_idx, "iter_idx": iter_idx}, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"t_idx": 0, "iter_idx": 0}

class Color(object):
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

def read_png(res): return np.array(PIL.Image.open(BytesIO(res)))

def match_color(object_mask, target_color, tolerance=3):
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3):
        min_val = max(0, target_color[c] - tolerance)
        max_val = min(255, target_color[c] + tolerance)
        match_region &= (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
    return match_region if match_region.sum() != 0 else None

def mask_to_box(mask):
    rows, cols = np.where(mask > 0)
    if len(rows) == 0: return np.array([0,0,0,0], dtype=float)
    return np.array([np.min(cols), np.min(rows), np.max(cols), np.max(rows)], dtype=float)

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else 0.0

def show_masks(image, mask, gt_mask, box_coords, ind, results_dir):
    base_vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    curr_iou = calculate_iou(mask, gt_mask)
    vis = base_vis.copy()
    
    mask_bool = mask > 0
    if mask_bool.any():
        color_layer = np.zeros_like(vis)
        color_layer[:] = [255, 144, 30]
        vis[mask_bool] = cv2.addWeighted(vis, 0.4, color_layer, 0.6, 0)[mask_bool]

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [cv2.approxPolyDP(cnt, epsilon=0.01, closed=True) for cnt in contours]
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)

    x1, y1, x2, y2 = box_coords.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    title = f"View {ind} | IoU: {curr_iou:.3f}"
    cv2.putText(vis, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(vis, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    save_path = os.path.join(results_dir, f"view_{ind:04d}.png")
    cv2.imwrite(save_path, vis)
    return curr_iou

def run_experiment(client, origin_x, origin_y, origin_z, radius, id2color, single_object, materials_dict):
    checkpoint = load_checkpoint()
    start_t = checkpoint["t_idx"]
    start_iter = checkpoint["iter_idx"]

    target_map = materials_dict
    search_space = list(itertools.product(range(90, 60, -10), range(0, 390, 30)))   # Change camera viewpoints as needed
    
    start_time = time.time()

    for t_idx, (_, t_path) in enumerate(target_map.items()):
        if t_idx < start_t: continue

        t_splits = t_path.split('/')
        t_name = t_splits[-1]
        t_category = t_splits[-2]
        print(f"\n>>> Processing Texture: {t_name} (Category: {t_category})")

        # Apply material to target object
        client.request(f"lych object set_material {single_object} {t_path} 0")

        # Setup Directories
        run_id = f"TEXTURE_{t_category}_{t_name}"
        run_dir = os.path.join(OUTPUT_BASE, run_id)
        renders_dir = os.path.join(run_dir, "renders")
        results_dir = os.path.join(run_dir, "results")
        masks_dir = os.path.join(run_dir, "masks")
        
        for d in [run_dir, renders_dir, results_dir, masks_dir]:
            os.makedirs(d, exist_ok=True)

        iou_store_path = os.path.join(run_dir, "run_ious.npy")
        if os.path.exists(iou_store_path):
            persistent_ious = np.load(iou_store_path).tolist()
        else:
            persistent_ious = [0.0] * len(search_space)

        for iter_idx, (az, pol) in enumerate(search_space):
            if t_idx == start_t and iter_idx < start_iter:
                continue
            
            print(f"--- Iteration [{iter_idx+1}/{len(search_space)}] | Az: {az} Pol: {pol} ---")

            # Camera Positioning
            rad_az, rad_pol = az * np.pi / 180, pol * np.pi / 180
            x = radius * np.sin(rad_az) * np.cos(rad_pol) + origin_x
            y = radius * np.sin(rad_az) * np.sin(rad_pol) + origin_y
            z = radius * np.cos(rad_az) + origin_z
            
            client.request(f'vset /camera/1/location {x} {y} {z}')
            client.request(f'vset /camera/1/rotation {az-90} {pol+180} 0')

            # Get GT Mask
            res = client.request('vget /camera/1/object_mask png')
            mask_raw = read_png(res)
            col = id2color[single_object]
            gt_mask = match_color(mask_raw, [col.R, col.G, col.B])

            if gt_mask is None or np.sum(gt_mask) == 0: 
                print(f"Skipping view {iter_idx}: Object not visible.")
                continue

            gt_box = mask_to_box(gt_mask)
            np.save(os.path.join(masks_dir, f"gt_{iter_idx:04d}.npy"), gt_mask)

            # Capture Render
            render_path = os.path.join(renders_dir, f'render_{iter_idx:04d}.png')
            client.request(f'vget /camera/1/lit {render_path}')

            # SAM Inference
            img_np = np.array(PIL.Image.open(render_path).convert("RGB"))
            original_size = img_np.shape[:2]
            input_img = resize_transform.apply_image(img_np)
            input_torch = torch.as_tensor(input_img, device=DEVICE).permute(2,0,1).contiguous()
            box_torch = torch.as_tensor(resize_transform.apply_boxes(gt_box.reshape(1,4), original_size), dtype=torch.float, device=DEVICE)

            with torch.no_grad():
                output = sam([{'image': input_torch, 'boxes': box_torch, 'original_size': original_size}], multimask_output=False)
                pred_mask = (output[0]['masks'][0, 0].cpu().numpy() > 0)

            # Serialization & Logging
            compressed = zlib.compress(pred_mask.astype(np.float16).tobytes())
            b64_logits = base64.b64encode(compressed).decode('utf-8')
            
            iou = show_masks(img_np, pred_mask, gt_mask, gt_box, iter_idx, results_dir)
            persistent_ious[iter_idx] = float(iou)
            np.save(iou_store_path, np.array(persistent_ious))

            entry = {
                "object": t_name,
                "category": t_category,
                "view_idx": iter_idx,
                "angles": {"azimuth": az, "polar": pol},
                "iou": float(iou),
                "logits": b64_logits,
                "shape": pred_mask.shape
            }
            with open(RESULTS_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            save_checkpoint(t_idx, iter_idx + 1)

        # Reset iteration for next texture
        start_iter = 0

        # Plot result for current texture
        plt.figure(figsize=(10, 5))
        plt.plot(persistent_ious, marker='o', color='orange')
        plt.title(f"IoU: {t_name}")
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, f"iou_graph_{run_id}.png"))
        plt.close()

    print(f"\n--- Search Finished in {time.time() - start_time:.2f} seconds ---")


if __name__ == '__main__':
    PORT_NUMBER = 9000  # Change as needed
    ip_address = ""
    client = unrealcv.Client(ip_address, PORT_NUMBER)
    client.connect()

    if not client.isconnected():
        print("Initial connection failed.")
        os._exit(1)

    client.request('vset /cameras/spawn')

    with open("/path/to/texture_paths.json", 'r') as f:
        materials_dict = json.load(f)
    
    single_object = "StaticMeshActor_1"     # Change to the target object's UE path name
    color_info = client.request(f'vget /object/{single_object}/color')
    id2color = {single_object: Color(color_info)}
    # Modify below parameter coordinates and distance as needed
    x, y, z = 730.0, -215.0, 30,0
    radius = 175
    run_experiment(client, x, y, z, radius, id2color, single_object, materials_dict)