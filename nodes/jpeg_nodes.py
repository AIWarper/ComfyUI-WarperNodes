import os
import uuid

import folder_paths
import numpy as np
from PIL import Image


class SaveJPEG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Warper Tools/Image"
    DESCRIPTION = "Save images as JPEG files and display them in the node."

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    def save_images(self, images, filename_prefix="ComfyUI", quality=95):
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix + self.prefix_append,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )
        results = []
        for batch_number, image in enumerate(images):
            pixels = np.clip(255.0 * image.detach().cpu().numpy(), 0, 255).astype(np.uint8)
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            # JPEG has no alpha channel; retain RGB and discard alpha if present.
            img = Image.fromarray(pixels).convert("RGB")
            batch_filename = filename.replace("%batch_num%", str(batch_number))
            file = f"{batch_filename}_{counter:05}_.jpg"
            img.save(
                os.path.join(full_output_folder, file),
                format="JPEG",
                quality=quality,
                subsampling=0,
                optimize=True,
            )
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1
        return {"ui": {"images": results}}


class PreviewJPEG(SaveJPEG):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            }
        }

    DESCRIPTION = "Preview JPEG images using temporary files instead of the output directory."

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + uuid.uuid4().hex
