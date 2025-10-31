# aiwarper-comfyui-warpernodes/__init__.py
from .nodes.wan_flow_nodes import FlowVisualizerNode_Warper # Removed FlowGetFlowNode, GetRaftFlowNode, FlowConfigNode
from .nodes.warper_nodes import (DWPoseScalerNode_Warper, MouthMaskFromPose_Warper, FacialPartMaskFromPose_Warper)
from .nodes.wan_video_batching_nodes import (
    SmartVideoBatcher,
    GetBatchByIndex,
    SmartOverlappingBatcher,
)
from .nodes.image_resolution_nodes import (
    PreprocessForTarget,
    CropAndRestore,
    AspectRatioMatchToBase,
    AspectRatioMatchToStandardResolution,
    AspectRatioResolution,
)

NODE_CLASS_MAPPINGS = {
    # DWPose Nodes
    "DWPoseScalerNode_Warper": DWPoseScalerNode_Warper,
    "MouthMaskFromPose_Warper": MouthMaskFromPose_Warper,
    "FacialPartMaskFromPose_Warper": FacialPartMaskFromPose_Warper,
    # Visual Looping Nodes
    "SmartVideoBatcher_Warper": SmartVideoBatcher,
    "GetBatchByIndex_Warper": GetBatchByIndex,
    "SmartOverlappingBatcher_Warper": SmartOverlappingBatcher,
    # MAPPINGS FOR FLOW NODES (Advanced) - REMOVED
    # "FlowGetFlow_Warper": FlowGetFlowNode, # Removed
    # "GetRaftFlow_Warper": GetRaftFlowNode, # Removed
    # "FlowConfig_Warper": FlowConfigNode,   # Removed
    # MAPPING FOR SIMPLIFIED FLOW NODE
    "FlowVisualizerNode_Warper": FlowVisualizerNode_Warper,
    # Image Resolution Nodes
    "PreprocessForTarget_Warper": PreprocessForTarget,
    "CropAndRestore_Warper": CropAndRestore,
    "AspectRatioMatchToBase_Warper": AspectRatioMatchToBase,
    "AspectRatioMatchToStandardResolution_Warper": AspectRatioMatchToStandardResolution,
    "AspectRatioResolution_Warper": AspectRatioResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # DWPose Nodes
    "DWPoseScalerNode_Warper": "DWPose Scaler (Warper)",
    "MouthMaskFromPose_Warper": "Mouth Mask from Pose (Warper)",
    "FacialPartMaskFromPose_Warper": "Facial Part Mask from Pose (Warper)",
    # Visual Looping Nodes
    "SmartVideoBatcher_Warper": "Smart Video Batcher (Warper)",
    "GetBatchByIndex_Warper": "Get Batch By Index (Warper)",
    "SmartOverlappingBatcher_Warper": "Smart Overlapping Batcher (Warper)",
    # DISPLAY NAMES FOR FLOW NODES (Advanced) - REMOVED
    # "FlowGetFlow_Warper": "Get UniMatch Flow (Warper)", # Removed
    # "GetRaftFlow_Warper": "Get RAFT Flow (Warper)",   # Removed
    # "FlowConfig_Warper": "Flow Config (Warper)",     # Removed
    # DISPLAY NAME FOR SIMPLIFIED FLOW NODE
    "FlowVisualizerNode_Warper": "Flow Visualizer (Warper)",
    # Image Resolution Nodes
    "PreprocessForTarget_Warper": "Preprocess for Target (Warper)",
    "CropAndRestore_Warper": "Crop and Restore (Warper)",
    "AspectRatioMatchToBase_Warper": "Aspect Ratio Match to Base (Warper)",
    "AspectRatioMatchToStandardResolution_Warper": "Aspect Ratio Match to Standard Resolution (Warper)",
    "AspectRatioResolution_Warper": "Aspect Ratio Resolution (Warper)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("Successfully loaded AIWarper Nodes by Comfy Resolved / AIWarper")