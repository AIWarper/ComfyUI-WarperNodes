# aiwarper-comfyui-warpernodes/__init__.py
from .nodes.warper_nodes import (DWPoseScalerNode_Warper, MouthMaskFromPose_Warper, FacialPartMaskFromPose_Warper)
from .nodes.wan_video_batching_nodes import (
    SmartVideoBatcher,
    GetBatchByIndex,
    SmartOverlappingBatcher, # <-- Added the new overlapping batcher
    # The following are placeholders for future nodes
    # IterativeLoopSetup,
    # ConditionalLoopInputSwitch,
    # IterativeLoopFeedback
)

NODE_CLASS_MAPPINGS = {
    # DWPose Nodes
    "DWPoseScalerNode_Warper": DWPoseScalerNode_Warper,
    "MouthMaskFromPose_Warper": MouthMaskFromPose_Warper,
    "FacialPartMaskFromPose_Warper": FacialPartMaskFromPose_Warper,
    # Visual Looping Nodes
    "SmartVideoBatcher_Warper": SmartVideoBatcher,
    "GetBatchByIndex_Warper": GetBatchByIndex,
    "SmartOverlappingBatcher_Warper": SmartOverlappingBatcher, # <-- Registered the new node class
    # "IterativeLoopSetup_Warper": IterativeLoopSetup,
    # "ConditionalLoopInputSwitch_Warper": ConditionalLoopInputSwitch,
    # "IterativeLoopFeedback_Warper": IterativeLoopFeedback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # DWPose Nodes
    "DWPoseScalerNode_Warper": "DWPose Scaler (Warper)",
    "MouthMaskFromPose_Warper": "Mouth Mask from Pose (Warper)",
    "FacialPartMaskFromPose_Warper": "Facial Part Mask from Pose (Warper)",
    # Visual Looping Nodes
    "SmartVideoBatcher_Warper": "Smart Video Batcher (Warper)",
    "GetBatchByIndex_Warper": "Get Batch By Index (Warper)",
    "SmartOverlappingBatcher_Warper": "Smart Overlapping Batcher (Warper)", # <-- Added the display name
    # "IterativeLoopSetup_Warper": "Iterative Loop Setup (Warper)",
    # "ConditionalLoopInputSwitch_Warper": "Conditional Loop Input Switch (Warper)",
    # "IterativeLoopFeedback_Warper": "Iterative Loop Feedback (Warper)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("Successfully loaded AIWarper Nodes by Comfy Resolved / AIWarper")