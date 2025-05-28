# ComfyUI/custom_nodes/ComfyUI_WarperNodes/__init__.py

# Import your node class(es) from other files in this package
from .warper_nodes import DWPoseScalerNode_Warper # Corrected import

# A dictionary that ComfyUI uses to know what nodes are available and how to construct them
NODE_CLASS_MAPPINGS = {
    "DWPoseScalerNode_Warper": DWPoseScalerNode_Warper,
    # Add other nodes from your pack here
}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPoseScalerNode_Warper": "DWPose Scaler (Warper)",
    # Add other display names here
}

# Optional: If you have a ./js folder with custom JavaScript for your nodes
# WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] # Removed WEB_DIRECTORY if not used

print("Successfully loaded Warper Nodes pack by Comfy Resolved") # Your custom message