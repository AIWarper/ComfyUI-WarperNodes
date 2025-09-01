# ComfyUI/custom_nodes/ComfyUI_WarperNodes/nodes/image_resolution_nodes.py

import torch
import numpy as np
from PIL import Image
import re

# Target resolutions for Kontext model
PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class PreprocessForTarget:
    """
    Preprocesses an image by resizing and padding it to fit the best-matching target resolution.
    Supports two padding types:
    - Mirror: Uses mirrored edges for padding to maintain visual continuity
    - Colored: Uses solid color padding (black, grey, or white)
    Outputs both the processed image and the data needed to reverse this process.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Create default resolution string from PREFERED_KONTEXT_RESOLUTIONS
        default_resolutions = "\n".join([f"{w}, {h}" for w, h in PREFERED_KONTEXT_RESOLUTIONS])
        
        return {
            "required": {
                "image": ("IMAGE",),
                "target_resolutions": ("STRING", {
                    "multiline": True, 
                    "default": default_resolutions
                }),
                "padding_type": (["mirror", "colored"], {"default": "mirror"}),
                "padding_color": (["black", "grey", "white"], {"default": "black"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "PREPROCESS_DATA")
    RETURN_NAMES = ("image", "preprocess_data")
    FUNCTION = "preprocess"
    CATEGORY = "Warper Tools/Preprocessing"
    
    def parse_resolutions(self, resolution_string):
        """Parse resolution string into list of (width, height) tuples."""
        resolutions = []
        lines = resolution_string.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try different parsing patterns
            # Pattern 1: (width, height)
            match = re.match(r'\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*', line)
            if match:
                resolutions.append((int(match.group(1)), int(match.group(2))))
                continue
                
            # Pattern 2: width, height
            match = re.match(r'\s*(\d+)\s*,\s*(\d+)\s*', line)
            if match:
                resolutions.append((int(match.group(1)), int(match.group(2))))
                continue
                
            # Pattern 3: width x height
            match = re.match(r'\s*(\d+)\s*x\s*(\d+)\s*', line)
            if match:
                resolutions.append((int(match.group(1)), int(match.group(2))))
                continue
                
            print(f"Warning: Could not parse resolution line: {line}")
            
        return resolutions
    
    def find_best_resolution(self, source_width, source_height, target_resolutions):
        """Find the target resolution with the closest aspect ratio to the source."""
        source_aspect = source_width / source_height
        best_resolution = None
        min_diff = float('inf')
        
        for width, height in target_resolutions:
            target_aspect = width / height
            diff = abs(source_aspect - target_aspect)
            
            if diff < min_diff:
                min_diff = diff
                best_resolution = (width, height)
                
        return best_resolution
    
    def create_mirrored_padding(self, image, target_width, target_height):
        """Create a padded image using mirrored edges."""
        img_width, img_height = image.size
        
        # Calculate padding needed
        pad_left = (target_width - img_width) // 2
        pad_right = target_width - img_width - pad_left
        pad_top = (target_height - img_height) // 2
        pad_bottom = target_height - img_height - pad_top
        
        # Convert to numpy for easier manipulation
        img_array = np.array(image)
        
        # Use numpy's pad with 'reflect' mode for mirrored edges
        padded_array = np.pad(
            img_array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='reflect'
        )
        
        # Convert back to PIL Image
        padded_image = Image.fromarray(padded_array.astype(np.uint8))
        
        return padded_image, pad_left, pad_top
    
    def create_colored_padding(self, image, target_width, target_height, color):
        """Create a padded image using solid color padding."""
        img_width, img_height = image.size
        
        # Calculate padding needed
        pad_left = (target_width - img_width) // 2
        pad_top = (target_height - img_height) // 2
        
        # Define color values
        color_values = {
            "black": (0, 0, 0),
            "grey": (128, 128, 128),
            "white": (255, 255, 255)
        }
        
        # Create new image with solid color background
        padded_image = Image.new('RGB', (target_width, target_height), color_values[color])
        
        # Paste the original image in the center
        padded_image.paste(image, (pad_left, pad_top))
        
        return padded_image, pad_left, pad_top
    
    def preprocess(self, image, target_resolutions, padding_type, padding_color):
        # Convert from ComfyUI tensor to PIL Image
        # ComfyUI images are [batch, height, width, channels] with values 0-1
        image_np = image[0].cpu().numpy()  # Get first image in batch
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np, mode='RGB')
        
        # Get original dimensions
        w_orig, h_orig = pil_image.size
        
        # Parse target resolutions
        resolutions = self.parse_resolutions(target_resolutions)
        if not resolutions:
            print("Warning: No valid resolutions parsed, using default")
            resolutions = PREFERED_KONTEXT_RESOLUTIONS
            
        # Find best matching resolution
        best_resolution = self.find_best_resolution(w_orig, h_orig, resolutions)
        w_target, h_target = best_resolution
        
        # Calculate scale ratio
        scale_ratio = min(w_target / w_orig, h_target / h_orig)
        
        # Calculate new dimensions
        w_new = round(w_orig * scale_ratio)
        h_new = round(h_orig * scale_ratio)
        
        # Resize image using Lanczos resampling
        resized_image = pil_image.resize((w_new, h_new), Image.Resampling.LANCZOS)
        
        # Create padded image based on padding type
        if padding_type == "mirror":
            padded_image, paste_x, paste_y = self.create_mirrored_padding(resized_image, w_target, h_target)
        else:  # colored
            padded_image, paste_x, paste_y = self.create_colored_padding(resized_image, w_target, h_target, padding_color)
        
        # Convert back to ComfyUI tensor format
        padded_np = np.array(padded_image).astype(np.float32) / 255.0
        padded_tensor = torch.from_numpy(padded_np).unsqueeze(0)  # Add batch dimension
        
        # Prepare preprocess data
        crop_box = (paste_x, paste_y, paste_x + w_new, paste_y + h_new)
        preprocess_data = {
            "original_size": (w_orig, h_orig),
            "crop_box": crop_box,
            "processed_size": (w_target, h_target),
            "scale_ratio": scale_ratio
        }
        
        return (padded_tensor, preprocess_data)


class CropAndRestore:
    """
    Crops the padding from a processed image and resizes it back to its original dimensions.
    Uses the preprocessing data from PreprocessForTarget node.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocess_data": ("PREPROCESS_DATA",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    CATEGORY = "Warper Tools/Preprocessing"
    
    def restore(self, image, preprocess_data):
        # Extract data
        w_orig, h_orig = preprocess_data["original_size"]
        crop_box = preprocess_data["crop_box"]
        
        # Convert from ComfyUI tensor to PIL Image
        image_np = image[0].cpu().numpy()  # Get first image in batch
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np, mode='RGB')
        
        # Crop the image to remove padding
        cropped_image = pil_image.crop(crop_box)
        
        # Resize back to original dimensions using Lanczos resampling
        restored_image = cropped_image.resize((w_orig, h_orig), Image.Resampling.LANCZOS)
        
        # Convert back to ComfyUI tensor format
        restored_np = np.array(restored_image).astype(np.float32) / 255.0
        restored_tensor = torch.from_numpy(restored_np).unsqueeze(0)  # Add batch dimension
        
        return (restored_tensor,)


class AspectRatioResolution:
    """
    Calculates width and height based on a selected aspect ratio and desired long edge resolution.
    The long edge is the larger dimension (width for landscape, height for portrait).
    """
    
    # Define aspect ratios as tuples of (width_ratio, height_ratio, display_name)
    ASPECT_RATIOS = [
        (21, 9, "21:9 (Ultra-Wide)"),
        (16, 9, "16:9 (Wide)"),
        (4, 3, "4:3 (Standard)"),
        (3, 2, "3:2 (Photo)"),
        (1, 1, "1:1 (Square)"),
        (2, 3, "2:3 (Portrait Photo)"),
        (3, 4, "3:4 (Portrait Standard)"),
        (9, 16, "9:16 (Portrait Wide)"),
        (9, 21, "9:21 (Portrait Ultra-Wide)"),
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        # Create display names for the dropdown
        aspect_ratio_options = [ratio[2] for ratio in cls.ASPECT_RATIOS]
        
        return {
            "required": {
                "aspect_ratio": (aspect_ratio_options, {"default": "16:9 (Wide)"}),
                "long_edge": ("INT", {
                    "default": 1920,
                    "min": 64,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate_resolution"
    CATEGORY = "Warper Tools/Resolution"
    
    def calculate_resolution(self, aspect_ratio, long_edge):
        # Find the matching aspect ratio tuple
        ratio_tuple = None
        for ratio in self.ASPECT_RATIOS:
            if ratio[2] == aspect_ratio:
                ratio_tuple = ratio
                break
        
        if not ratio_tuple:
            raise ValueError(f"Unknown aspect ratio: {aspect_ratio}")
        
        width_ratio, height_ratio, _ = ratio_tuple
        
        # Calculate dimensions based on which edge is longer
        if width_ratio >= height_ratio:
            # Landscape or square - width is the long edge (or equal)
            width = long_edge
            height = round(long_edge * height_ratio / width_ratio)
        else:
            # Portrait - height is the long edge
            height = long_edge
            width = round(long_edge * width_ratio / height_ratio)
        
        # Ensure dimensions are even numbers (often required for video encoding)
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1
        
        return (width, height)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PreprocessForTarget_Warper": PreprocessForTarget,
    "CropAndRestore_Warper": CropAndRestore,
    "AspectRatioResolution_Warper": AspectRatioResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreprocessForTarget_Warper": "Preprocess for Target (Warper)",
    "CropAndRestore_Warper": "Crop and Restore (Warper)",
    "AspectRatioResolution_Warper": "Aspect Ratio Resolution (Warper)",
}