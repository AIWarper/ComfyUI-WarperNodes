from .nodes.jpeg_nodes import SaveJPEG, PreviewJPEG

NODE_CLASS_MAPPINGS = {
    "SaveJPEG_Warper": SaveJPEG,
    "PreviewJPEG_Warper": PreviewJPEG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveJPEG_Warper": "Save JPEG (Warper)",
    "PreviewJPEG_Warper": "Preview JPEG (Warper)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
