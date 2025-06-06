# aiwarper-comfyui-warpernodes/__init__.py

from .nodes.warper_nodes import (
    DWPoseScalerNode_Warper, MouthMaskFromPose_Warper, FacialPartMaskFromPose_Warper
)
from .nodes.wan_video_batching_nodes import (
    SmartVideoBatcher, IterativeLoopSetup, IterativeLoopFeedback,
    ConditionalLoopInputSwitch, ToAny, FromAny # ADD THE NEW CONVERTERS
)

NODE_CLASS_MAPPINGS = {
    "DWPoseScalerNode_Warper": DWPoseScalerNode_Warper,
    "MouthMaskFromPose_Warper": MouthMaskFromPose_Warper,
    "FacialPartMaskFromPose_Warper": FacialPartMaskFromPose_Warper,
    
    "SmartVideoBatcher_Wan": SmartVideoBatcher,
    "IterativeLoopSetup_Wan": IterativeLoopSetup,
    "IterativeLoopFeedback_Wan": IterativeLoopFeedback,
    "ConditionalLoopInputSwitch_Wan": ConditionalLoopInputSwitch,
    "ToAny_Wan": ToAny,                 # ADD THIS
    "FromAny_Wan": FromAny,               # ADD THIS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPoseScalerNode_Warper": "DWPose Scaler (Warper)",
    "MouthMaskFromPose_Warper": "Mouth Mask from Pose (Warper)",
    "FacialPartMaskFromPose_Warper": "Facial Part Mask from Pose (Warper)",
    
    "SmartVideoBatcher_Wan": "Smart Video Batcher (Wan)",
    "IterativeLoopSetup_Wan": "Iterative Loop Setup (Wan)",
    "IterativeLoopFeedback_Wan": "Iterative Loop Feedback (Wan)",
    "ConditionalLoopInputSwitch_Wan": "Conditional Loop Input Switch (Wan)",
    "ToAny_Wan": "Convert To Any (Loop Breaker)", # ADD THIS
    "FromAny_Wan": "Convert From Any (Loop Breaker)",# ADD THIS
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("Successfully loaded AIWarper Nodes by Comfy Resolved / AIWarper")