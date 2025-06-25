# aiwarper-comfyui-warpernodes/nodes/wan_flow_nodes.py
# This file combines the optical flow nodes from ComfyUI-VeeVee for integration.

import os
import torch
import torch.nn.functional as F
from einops import rearrange
from folder_paths import models_dir

# --- Dependencies for UniMatch Flow ---
# UniMatch related imports and class are removed

# --- Dependencies for RAFT Flow ---
try:
    from ..sea_raft.custom import get_model as get_raft_model
except ImportError:
    print("Warning: SEA-RAFT not found. The 'Get RAFT Flow' and 'Flow Visualizer' nodes will not work with RAFT.")
    print("Please ensure the 'sea_raft' directory is present in the 'aiwarper-comfyui-warpernodes' folder.")
    get_raft_model = None

# --- Shared Utility Imports ---
from ..utils.flow_viz import flow_to_color
from ..utils.flow_utils import get_full_frames_trajectories, flow_warp, forward_backward_consistency_check


# --- Model Paths ---
# UNIMATCH_PATH is removed
RAFT_PATH = os.path.join(models_dir, 'raft')
# os.makedirs(UNIMATCH_PATH, exist_ok=True) # Removed
os.makedirs(RAFT_PATH, exist_ok=True)

# get_unimatch_files() is removed

def get_raft_files():
    if not os.path.exists(RAFT_PATH):
        return []
    return os.listdir(RAFT_PATH)

def get_all_flow_models():
    # Helper to combine lists for the new visualizer node
    # Now only returns RAFT files
    return get_raft_files()

# --- Helper Classes and Functions ---

class InputPadder:
    """ Pads images such that dimensions are divisible by a number (e.g., 8)."""
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [0, pad_wd, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def get_backward_occlusions(images, forward_flows, backward_flows):
    _, backward_occlusions = forward_backward_consistency_check(forward_flows, backward_flows)
    reshuffle_list = list(range(1,len(images)))+[0]
    warped_image1 = flow_warp(images, backward_flows)
    backward_occlusions = torch.clamp(backward_occlusions + (abs(images[reshuffle_list]-warped_image1).mean(dim=1)>255*0.25).float(), 0 ,1)
    return backward_occlusions

def get_trajectories(images, backward_flows, backward_occlusions, scale=1):
    images_rearranged = rearrange(images, 'b h w c -> b c h w')
    forward_trajectory, backward_trajectory, attn_masks = get_full_frames_trajectories(backward_flows, backward_occlusions, images_rearranged, scale=8.0 * scale)
    trajectories = {
        'forward_trajectory': forward_trajectory,
        'backward_trajectory': backward_trajectory,
        'attn_masks': attn_masks
    }
    return trajectories

# --- Node Class Definitions ---

# FlowGetFlowNode class is removed as it was UniMatch specific.

class GetRaftFlowNode:
    """
    Calculates optical flow using SEA-RAFT and returns full trajectory data and visualizations.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "checkpoint": (get_raft_files(),),
                "flow_type": (['SD15', 'SD15_Full', 'SDXL'],),
                "direction": (['forward', 'backward', 'both'],)
            }
        }

    RETURN_TYPES = ("FLOW", "IMAGE", "IMAGE")
    RETURN_NAMES = ("flow_data", "forward_flow_viz", "backward_flow_viz")
    FUNCTION = "process"
    CATEGORY = "Warper Tools/Flow"

    @torch.no_grad()
    def pred_flows(self, flow_model, images):
        images_cuda = rearrange(images, 'b h w c -> b c h w').cuda() * 255.0
        
        # RAFT often benefits from padding too, though its requirements might be less strict
        # For consistency and to avoid potential issues, let's pad for RAFT as well.
        # RAFT models are often trained on inputs divisible by 8.
        padder = InputPadder(images_cuda.shape, divis_by=8) 
        images_cuda_padded = padder.pad(images_cuda)[0]
        
        reshuffle_list = list(range(1,len(images_cuda_padded)))+[0]
        images_r_padded = images_cuda_padded[reshuffle_list]
        forward_flows, backward_flows = [], []

        for i in range(len(images_cuda_padded)):
            forward_flow, _ = flow_model(images_cuda_padded[i:i+1], images_r_padded[i:i+1])
            forward_flows.append(padder.unpad(forward_flow).cpu()) # Unpad flow

        for i in range(len(images_cuda_padded)):
            backward_flow, _= flow_model(images_r_padded[i:i+1], images_cuda_padded[i:i+1])
            backward_flows.append(padder.unpad(backward_flow).cpu()) # Unpad flow

        forward_flows = torch.cat(forward_flows, dim=0)
        backward_flows = torch.cat(backward_flows, dim=0)
        
        # Occlusion check should happen on original, unpadded images
        images_cpu_chw = rearrange(images, 'b h w c -> b c h w') * 255.0
        forward_occlusions = get_backward_occlusions(images_cpu_chw, backward_flows, forward_flows)
        backward_occlusions = get_backward_occlusions(images_cpu_chw, forward_flows, backward_flows)
        
        return {
            'forward': forward_flows,
            'backward': backward_flows,
            'forward_occlusions': forward_occlusions,
            'backward_occlusions': backward_occlusions
        }

    def process(self, images, checkpoint, flow_type, direction):
        if get_raft_model is None:
             raise ImportError("SEA-RAFT model could not be imported. Please ensure the 'sea_raft' directory is correctly placed.")

        checkpoint_path = os.path.join(RAFT_PATH, checkpoint)
        flow_model = get_raft_model(checkpoint_path).to('cuda')
        flow_model.eval()
        
        pred = self.pred_flows(flow_model, images)
        del flow_model
        
        forward_flow_imgs = flow_to_color(pred['forward'])
        forward_flow_imgs = rearrange(forward_flow_imgs, 'b c h w -> b h w c') / 255.

        backward_flow_images = flow_to_color(pred['backward'])
        backward_flow_images = rearrange(backward_flow_images, 'b c h w -> b h w c') / 255.

        get_forward_flow = direction == 'forward' or direction == 'both'
        get_backward_flow = direction == 'backward' or direction == 'both'
        forward_flows, backward_flows = [], []
        
        scales = []
        if flow_type == 'SD15': scales = [1]
        elif flow_type == 'SD15_Full': scales = [1, 2, 4]
        elif flow_type == 'SDXL': scales = [2, 4]
        
        for scale in scales:
            if get_forward_flow:
                forward_flows.append(get_trajectories(images, pred['backward'], pred['backward_occlusions'], scale))
            if get_backward_flow:
                backward_flows.append(get_trajectories(images, pred['forward'], pred['forward_occlusions'], scale))
            
        trajectory_data = {'forward_flows': forward_flows, 'backward_flows': backward_flows, 'direction': direction}
        torch.cuda.empty_cache()
        return (trajectory_data, forward_flow_imgs, backward_flow_images)


class FlowConfigNode:
    """
    Packages the output from a flow calculation node into a configuration object.
    Mainly for compatibility with systems that expect this specific format.
    """
    @classmethod
    def INPUT_TYPES(s):
        MAP_TYPES = ['full', 'inner', 'outer', 'none', 'input', 'output']
        return {"required": { 
            "flow": ("FLOW",),
            "targets": (MAP_TYPES,),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("FLOW_CONFIG",)
    FUNCTION = "build"
    CATEGORY = "Warper Tools/Flow"

    def build(self, flow, targets, start_percent, end_percent):
        config = {'targets': targets, 'flow': flow, 'start_percent': start_percent, 'end_percent': end_percent}
        return (config,)


# --- Simplified Node (Now RAFT-only) ---

class FlowVisualizerNode_Warper:
    """
    Calculates optical flow using RAFT and returns ONLY the visual representation as frames.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                # "model_type": (["RAFT"],), # Only RAFT is now an option
                "checkpoint": (get_raft_files(),), # Use get_raft_files directly
                "direction": (['forward', 'backward', 'both'], {"default": "both"})
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("forward_flow_viz", "backward_flow_viz")
    FUNCTION = "process"
    CATEGORY = "Warper Tools/Flow"

    # _pred_unimatch_flows method is removed

    @torch.no_grad()
    def _pred_raft_flows(self, flow_model, images, get_forward, get_backward):
        images_cuda = rearrange(images, 'b h w c -> b c h w').cuda() * 255.0
        
        # Pad for RAFT (typically divisible by 8)
        padder = InputPadder(images_cuda.shape, divis_by=8)
        images_cuda_padded = padder.pad(images_cuda)[0]
        
        reshuffle_list = list(range(1,len(images_cuda_padded)))+[0]
        images_r_padded = images_cuda_padded[reshuffle_list]
        forward_flows, backward_flows = [], []
        
        if get_forward:
            for i in range(len(images_cuda_padded)):
                forward_flow, _ = flow_model(images_cuda_padded[i:i+1], images_r_padded[i:i+1])
                forward_flows.append(padder.unpad(forward_flow).cpu()) # Unpad flow

        if get_backward:
            for i in range(len(images_cuda_padded)):
                backward_flow, _ = flow_model(images_r_padded[i:i+1], images_cuda_padded[i:i+1])
                backward_flows.append(padder.unpad(backward_flow).cpu()) # Unpad flow

        return torch.cat(forward_flows, dim=0) if get_forward and forward_flows else None, \
               torch.cat(backward_flows, dim=0) if get_backward and backward_flows else None


    def process(self, images, checkpoint, direction): # model_type removed from parameters
        get_forward = direction in ['forward', 'both']
        get_backward = direction in ['backward', 'both']
        
        forward_flow_vectors, backward_flow_vectors = None, None
        
        # Directly use RAFT
        if get_raft_model is None: raise ImportError("SEA-RAFT model could not be imported.")
        if checkpoint not in get_raft_files(): raise FileNotFoundError(f"RAFT model '{checkpoint}' not found in {RAFT_PATH}")
        
        checkpoint_path = os.path.join(RAFT_PATH, checkpoint)
        flow_model = get_raft_model(checkpoint_path).to('cuda')
        flow_model.eval()
        forward_flow_vectors, backward_flow_vectors = self._pred_raft_flows(flow_model, images, get_forward, get_backward)

        del flow_model
        torch.cuda.empty_cache()

        h, w = images.shape[1], images.shape[2]
        # Create a single black frame if a flow direction is not requested or fails
        empty_img_tensor = torch.zeros((images.shape[0] if images.shape[0] > 0 else 1, h, w, 3), dtype=torch.float32, device="cpu")
        
        forward_viz = flow_to_color(forward_flow_vectors) if forward_flow_vectors is not None else None
        backward_viz = flow_to_color(backward_flow_vectors) if backward_flow_vectors is not None else None

        final_forward_viz = rearrange(forward_viz, 'b c h w -> b h w c') / 255. if forward_viz is not None else empty_img_tensor
        final_backward_viz = rearrange(backward_viz, 'b c h w -> b h w c') / 255. if backward_viz is not None else empty_img_tensor

        return (final_forward_viz, final_backward_viz)