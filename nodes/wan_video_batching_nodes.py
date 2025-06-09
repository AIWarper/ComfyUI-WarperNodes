# aiwarper-comfyui-warpernodes/nodes/wan_video_batching_nodes.py

import torch

class SmartVideoBatcher:
    """
    A node that takes a batch of images (frames) and splits them into smaller,
    user-defined batches. It applies special padding to the final batch to ensure
    its length satisfies the formula (n-1) % 4 == 0.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch_length": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE_BATCHES",)
    RETURN_NAMES = ("image_batches",)
    FUNCTION = "batch_images"
    CATEGORY = "AIWarper/Looping"

    def batch_images(self, image: torch.Tensor, batch_length: int):
        total_frames = image.shape[0]
        
        print(f"[SmartVideoBatcher] Received {total_frames} frames with a desired batch length of {batch_length}.")

        if total_frames == 0:
            print("[SmartVideoBatcher] Warning: Input contains 0 frames. Returning empty list.")
            return ([],)

        if batch_length <= 0:
            print(f"[SmartVideoBatcher] Warning: batch_length ({batch_length}) is not positive. Returning the original batch as a single item list.")
            return ([image],)

        batches = []
        
        num_full_batches = total_frames // batch_length
        for i in range(num_full_batches):
            start_index = i * batch_length
            end_index = start_index + batch_length
            batch = image[start_index:end_index]
            batches.append(batch)
        
        print(f"[SmartVideoBatcher] Created {len(batches)} full-sized batch(es).")

        remaining_frames_count = total_frames % batch_length
        if remaining_frames_count > 0:
            start_of_remainder = num_full_batches * batch_length
            remaining_batch = image[start_of_remainder:]
            
            print(f"[SmartVideoBatcher] Handling remaining {remaining_frames_count} frames.")

            last_frame_to_duplicate = remaining_batch[-1].unsqueeze(0)
            current_length = remaining_frames_count
            target_length = current_length
            
            while (target_length - 1) % 4 != 0:
                target_length += 1
            
            num_to_pad = target_length - current_length

            if num_to_pad > 0:
                print(f"[SmartVideoBatcher] Padding last batch from {current_length} to {target_length} frames (adding {num_to_pad} duplicates of the last frame).")
                padding = last_frame_to_duplicate.repeat(num_to_pad, 1, 1, 1)
                padded_batch = torch.cat([remaining_batch, padding], dim=0)
                batches.append(padded_batch)
            else:
                print(f"[SmartVideoBatcher] No padding needed for the last batch of {current_length} frames.")
                batches.append(remaining_batch)
        
        total_output_frames = sum(b.shape[0] for b in batches)
        print(f"[SmartVideoBatcher] Finished. Outputting {len(batches)} batches with a total of {total_output_frames} frames.")
        
        return (batches,)

# +++ NEW NODE ADDED BELOW +++

class GetBatchByIndex:
    """
    Selects a single batch of images from a list of batches using an index.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_batches": ("IMAGE_BATCHES",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_batch"
    CATEGORY = "AIWarper/Looping"

    def get_batch(self, image_batches: list, index: int):
        # Check if the list of batches is empty
        if not image_batches:
            print("[GetBatchByIndex] Warning: Input 'image_batches' is empty. Returning an empty tensor.")
            # Return an empty tensor with the correct number of dimensions but zero size.
            # This avoids errors in downstream nodes expecting a 4D tensor.
            return (torch.empty(0, 1, 1, 3, dtype=torch.float32, device='cpu'),)

        # Check if the index is valid
        if index >= len(image_batches):
            print(f"[GetBatchByIndex] Error: Index {index} is out of bounds. The list has {len(image_batches)} batches. Returning the last available batch.")
            # Return the last batch to prevent crashing the workflow
            return (image_batches[-1],)
        
        print(f"[GetBatchByIndex] Selecting batch at index {index}.")
        return (image_batches[index],)


# In the future, other looping nodes will be added here.
# For example:
# class IterativeLoopSetup:
#     pass
# class ConditionalLoopInputSwitch:
#     pass
# class IterativeLoopFeedback:
#     pass