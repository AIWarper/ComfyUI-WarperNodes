# aiwarper-comfyui-warpernodes/nodes/wan_video_batching_nodes.py

import torch
import numpy as np

# ADD THESE TWO NEW CLASSES TO THE wan_video_batching_nodes.py FILE

class ToAny:
    """Converts any input to a generic ANY type to break frontend loop detection."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"data": ("*",)}}
    RETURN_TYPES = ("*",)
    FUNCTION = "convert"
    CATEGORY = "WarperNodes/WanVideoBatching/_Utils"

    def convert(self, data):
        return (data,)

class FromAny:
    """Converts a generic ANY type back to a specified primitive type."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*",),
                "target_type": (["LOOP_CONTEXT", "IMAGE", "INT"],), # Add other types if needed
            }
        }
    RETURN_TYPES = ("*",)
    FUNCTION = "convert"
    CATEGORY = "WarperNodes/WanVideoBatching/_Utils"

    def convert(self, data, target_type):
        # This node is mostly for the frontend; the backend just passes the data through.
        # The RETURN_TYPES being '*' lets it connect to anything.
        # We manually set the output type in the node definition to guide the user.
        self.RETURN_TYPES = (target_type,)
        return (data,)

class SmartVideoBatcher:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "preprocessed_video_frames": ("IMAGE",),
                "batch_length": ("INT", {"default": 81, "min": 5, "max": 512, "step": 1}),
            }}
    RETURN_TYPES = ("CONTROL_FRAMES_LIST",)
    RETURN_NAMES = ("control_frames_list",)
    FUNCTION = "batch_video"
    CATEGORY = "WarperNodes/WanVideoBatching"
    def batch_video(self, preprocessed_video_frames, batch_length):
        if preprocessed_video_frames is None or preprocessed_video_frames.ndim != 4 or preprocessed_video_frames.shape[0] == 0:
            return ([],)
        num_total_frames = preprocessed_video_frames.shape[0]
        L_ctrl = batch_length - 1
        if L_ctrl <= 0: raise ValueError(f"batch_length ({batch_length}) must be > 1.")
        control_frames_list = []
        current_idx = 0
        while current_idx < num_total_frames:
            remaining_frames = num_total_frames - current_idx
            if remaining_frames >= L_ctrl:
                segment = preprocessed_video_frames[current_idx : current_idx + L_ctrl]
                control_frames_list.append(segment)
                current_idx += L_ctrl
            else:
                remainder_segment = preprocessed_video_frames[current_idx:]
                current_model_input_len = 1 + remainder_segment.shape[0]
                padded_segment = remainder_segment
                while (current_model_input_len - 1) % 4 != 0:
                    if padded_segment.shape[0] == 0: break
                    padded_segment = torch.cat((padded_segment, padded_segment[-1].unsqueeze(0)), dim=0)
                    current_model_input_len += 1
                control_frames_list.append(padded_segment)
                break
        print(f"SmartVideoBatcher: Prepared {len(control_frames_list)} pre-processed control frame segments.")
        return (control_frames_list,)


class IterativeLoopSetup:
    """
    Initializes an iterative loop. Takes the full list of control frames and the
    user's initial start frame, and outputs the data for the first iteration, including
    an empty tensor to begin the stitched_frames accumulation.
    """
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "control_frames_list": ("CONTROL_FRAMES_LIST",),
                "first_start_frame": ("IMAGE",),
            }}
    # ADDED a new output: initial_stitched_frames
    RETURN_TYPES = ("IMAGE", "IMAGE", "LOOP_CONTEXT", "IMAGE") 
    RETURN_NAMES = ("initial_start_frame", "initial_control_frames", "loop_context", "initial_stitched_frames")
    FUNCTION = "setup_loop"
    CATEGORY = "WarperNodes/WanVideoBatching"

    def setup_loop(self, control_frames_list, first_start_frame):
        # Get the shape from the start frame to create a correctly dimensioned empty tensor
        B, H, W, C = first_start_frame.shape
        # This creates an image batch with 0 frames, which is the correct starting point for our accumulator
        initial_stitched_frames = torch.empty((0, H, W, C), dtype=first_start_frame.dtype, device="cpu")

        if not control_frames_list:
            print("IterativeLoopSetup: Warning - Empty control_frames_list. Loop will not run.")
            return (first_start_frame, torch.empty((0, H, W, C)), {"remaining_segments": [], "index": 0, "total": 0}, initial_stitched_frames)

        first_control_segment = control_frames_list[0]
        remaining_segments = control_frames_list[1:]
        
        loop_context = {
            "remaining_segments": remaining_segments,
            "index": 0,
            "total": len(control_frames_list)
        }
        
        return (first_start_frame, first_control_segment, loop_context, initial_stitched_frames)


class ConditionalLoopInputSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "iteration_index": ("INT", {"forceInput": True}),
                "initial_input": ("*",), 
                "feedback_input": ("*",),
            }}
    RETURN_TYPES = ("*",)
    FUNCTION = "switch_input"
    CATEGORY = "WarperNodes/WanVideoBatching/_Utils"
    def switch_input(self, iteration_index, initial_input, feedback_input):
        if iteration_index == 0:
            print(f"ConditionalSwitch: Iteration 0, passing initial_input.")
            return (initial_input,)
        else:
            print(f"ConditionalSwitch: Iteration {iteration_index}, passing feedback_input.")
            return (feedback_input,)


class IterativeLoopFeedback:
    # THIS IS THE CRUCIAL FIX.
    # This tells the ComfyUI frontend that this node is a "terminal" node
    # for the purpose of loop detection, preventing the "not submitting workflow" error.
    OUTPUT_NODE = True 
    
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
                "loop_context": ("LOOP_CONTEXT",),
                "processed_frames": ("IMAGE",),
            },
            "optional": {
                "stitched_frames_feedback": ("IMAGE",)
            }
        }
        
    RETURN_TYPES = ("IMAGE", "IMAGE", "LOOP_CONTEXT", "IMAGE", "INT")
    RETURN_NAMES = ("next_start_frame", "next_control_frames", "loop_context_out", "stitched_frames", "iteration_index")
    FUNCTION = "feedback_loop"
    CATEGORY = "WarperNodes/WanVideoBatching"

    def feedback_loop(self, loop_context, processed_frames, stitched_frames_feedback=None):
        current_index = loop_context.get("index", 0)
        total_iterations = loop_context.get("total", 0)
        remaining_segments = loop_context.get("remaining_segments", [])

        if stitched_frames_feedback is None:
            current_stitched_frames = processed_frames
        else:
            frames_to_add = processed_frames[1:] if processed_frames.shape[0] > 1 else processed_frames
            if frames_to_add.shape[0] > 0:
                 current_stitched_frames = torch.cat((stitched_frames_feedback, frames_to_add), dim=0)
            else:
                 current_stitched_frames = stitched_frames_feedback

        if not remaining_segments:
            print(f"IterativeLoopFeedback: Loop finished after {total_iterations} iterations.")
            empty_image = torch.empty((0,1,1,3))
            # Return empty data for the loop feedback outputs to terminate the loop
            return (empty_image, empty_image, {"remaining_segments": [], "index": -1, "total": total_iterations}, current_stitched_frames, current_index)
        
        next_control_frames = remaining_segments[0]
        next_remaining_segments = remaining_segments[1:]
        next_start_frame = processed_frames[-1].unsqueeze(0)
        
        next_loop_context = {
            "remaining_segments": next_remaining_segments,
            "index": current_index + 1,
            "total": total_iterations
        }
        
        print(f"IterativeLoopFeedback: Completed iteration {current_index + 1}/{total_iterations}. Preparing for next.")
        return (next_start_frame, next_control_frames, next_loop_context, current_stitched_frames, current_index)