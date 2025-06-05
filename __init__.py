# ComfyUI/custom_nodes/ComfyUI_WarperNodes/__init__.py

# Import your node class(es) from other files in this package
from .warper_nodes import (
    DWPoseScalerNode_Warper, 
    MouthMaskFromPose_Warper,
    FacialPartMaskFromPose_Warper
)

# A dictionary that ComfyUI uses to know what nodes are available and how to construct them
NODE_CLASS_MAPPINGS = {
    "DWPoseScalerNode_Warper": DWPoseScalerNode_Warper,
    "MouthMaskFromPose_Warper": MouthMaskFromPose_Warper,
    "FacialPartMaskFromPose_Warper": FacialPartMaskFromPose_Warper,
}

# A dictionary that ComfyUI uses to display the node names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPoseScalerNode_Warper": "DWPose Scaler (Warper)",
    "MouthMaskFromPose_Warper": "Mouth Mask from Pose (Warper)",
    "FacialPartMaskFromPose_Warper": "Facial Part Mask from Pose (Warper)",
}

# Optional: If you have a ./js folder with custom JavaScript for your nodes
# WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] # Removed WEB_DIRECTORY if not used

print("Successfully loaded Warper Nodes pack by Comfy Resolved") # Your custom message