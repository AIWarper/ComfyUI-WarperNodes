# ComfyUI/custom_nodes/ComfyUI_WarperNodes/nodes/text_file_iterator_nodes.py

import os
import glob
import random
from typing import Tuple

class TextFileIterator:
    """
    Load text files from a directory with different iteration modes.
    Supports fixed index, incremental (cycling), and randomized access.
    """
    
    # Class variable to store the last used index for incremental mode
    _incremental_indices = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to directory containing .txt files"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "mode": (["fixed", "incremental", "randomized"], {
                    "default": "fixed"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_content",)
    FUNCTION = "load_text_file"
    CATEGORY = "Warper Tools/Text"
    
    def load_text_file(self, directory_path: str, index: int, mode: str) -> Tuple[str]:
        """
        Load a text file based on the specified mode.
        
        Args:
            directory_path: Path to the directory containing .txt files
            index: Index value (used differently based on mode)
            mode: "fixed", "incremental", or "randomized"
            
        Returns:
            Tuple containing the text content
        """
        
        # Validate directory path
        if not directory_path:
            return ("",)
        
        # Expand user path and make absolute
        directory_path = os.path.expanduser(directory_path)
        if not os.path.isabs(directory_path):
            directory_path = os.path.abspath(directory_path)
        
        if not os.path.exists(directory_path):
            print(f"[TextFileIterator] Directory not found: {directory_path}")
            return ("",)
        
        if not os.path.isdir(directory_path):
            print(f"[TextFileIterator] Path is not a directory: {directory_path}")
            return ("",)
        
        # Find all .txt files in the directory
        txt_files = sorted(glob.glob(os.path.join(directory_path, "*.txt")))
        
        if not txt_files:
            print(f"[TextFileIterator] No .txt files found in: {directory_path}")
            return ("",)
        
        total_files = len(txt_files)
        
        # Determine which file to load based on mode
        if mode == "fixed":
            # Use the index directly, with bounds checking
            if index < 0 or index >= total_files:
                print(f"[TextFileIterator] Index {index} out of range (0-{total_files-1})")
                return ("",)
            file_index = index
            
        elif mode == "incremental":
            # Get the last used index for this directory
            if directory_path not in self._incremental_indices:
                self._incremental_indices[directory_path] = 0
            
            # Get current index and increment for next time
            file_index = self._incremental_indices[directory_path]
            self._incremental_indices[directory_path] = (file_index + 1) % total_files
            
        elif mode == "randomized":
            # Use index as seed for reproducible randomization
            random.seed(index)
            file_index = random.randint(0, total_files - 1)
            # Reset seed to avoid affecting other random operations
            random.seed()
        
        else:
            print(f"[TextFileIterator] Unknown mode: {mode}")
            return ("",)
        
        # Load the selected file
        txt_file = txt_files[file_index]
        file_name = os.path.basename(txt_file)
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"[TextFileIterator] Loaded {file_name} (mode: {mode}, index: {file_index})")
                return (content,)
        except Exception as e:
            print(f"[TextFileIterator] Error reading file {txt_file}: {e}")
            return ("",)


class TextFileIteratorWithInfo:
    """
    Enhanced version of TextFileIterator that also outputs file information.
    """
    
    # Class variable to store the last used index for incremental mode
    _incremental_indices = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to directory containing .txt files"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "mode": (["fixed", "incremental", "randomized"], {
                    "default": "fixed"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("text_content", "file_name", "file_index", "total_files")
    FUNCTION = "load_text_file_with_info"
    CATEGORY = "Warper Tools/Text"
    
    def load_text_file_with_info(self, directory_path: str, index: int, mode: str) -> Tuple[str, str, int, int]:
        """
        Load a text file with additional information about the file and directory.
        
        Args:
            directory_path: Path to the directory containing .txt files
            index: Index value (used differently based on mode)
            mode: "fixed", "incremental", or "randomized"
            
        Returns:
            Tuple containing (text content, file name, file index, total files)
        """
        
        # Validate directory path
        if not directory_path:
            return ("", "", 0, 0)
        
        # Expand user path and make absolute
        directory_path = os.path.expanduser(directory_path)
        if not os.path.isabs(directory_path):
            directory_path = os.path.abspath(directory_path)
        
        if not os.path.exists(directory_path):
            print(f"[TextFileIteratorWithInfo] Directory not found: {directory_path}")
            return ("", "", 0, 0)
        
        if not os.path.isdir(directory_path):
            print(f"[TextFileIteratorWithInfo] Path is not a directory: {directory_path}")
            return ("", "", 0, 0)
        
        # Find all .txt files in the directory
        txt_files = sorted(glob.glob(os.path.join(directory_path, "*.txt")))
        
        if not txt_files:
            print(f"[TextFileIteratorWithInfo] No .txt files found in: {directory_path}")
            return ("", "", 0, 0)
        
        total_files = len(txt_files)
        
        # Determine which file to load based on mode
        if mode == "fixed":
            # Use the index directly, with bounds checking
            if index < 0 or index >= total_files:
                print(f"[TextFileIteratorWithInfo] Index {index} out of range (0-{total_files-1})")
                return ("", "", index, total_files)
            file_index = index
            
        elif mode == "incremental":
            # Get the last used index for this directory
            if directory_path not in self._incremental_indices:
                self._incremental_indices[directory_path] = 0
            
            # Get current index and increment for next time
            file_index = self._incremental_indices[directory_path]
            self._incremental_indices[directory_path] = (file_index + 1) % total_files
            
        elif mode == "randomized":
            # Use index as seed for reproducible randomization
            random.seed(index)
            file_index = random.randint(0, total_files - 1)
            # Reset seed to avoid affecting other random operations
            random.seed()
        
        else:
            print(f"[TextFileIteratorWithInfo] Unknown mode: {mode}")
            return ("", "", 0, total_files)
        
        # Load the selected file
        txt_file = txt_files[file_index]
        file_name = os.path.basename(txt_file)
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"[TextFileIteratorWithInfo] Loaded {file_name} (mode: {mode}, index: {file_index}/{total_files-1})")
                return (content, file_name, file_index, total_files)
        except Exception as e:
            print(f"[TextFileIteratorWithInfo] Error reading file {txt_file}: {e}")
            return ("", file_name, file_index, total_files)