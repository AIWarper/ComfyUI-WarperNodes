# ComfyUI/custom_nodes/ComfyUI_WarperNodes/nodes/prompt_loader_nodes.py

import os
import glob
import torch
from typing import List, Tuple

class LoadPromptsFromDirectory:
    """
    Load text prompts from a directory with options to skip files and limit the number of loaded prompts.
    Designed to work with incremental integer nodes for sequential access through prompts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to directory containing .txt files"
                }),
                "skip_first_files": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "load_cap": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompts", "total_loaded", "file_names")
    FUNCTION = "load_prompts"
    CATEGORY = "Warper Tools/Prompts"
    OUTPUT_IS_LIST = (True, False, True)
    
    def load_prompts(self, directory_path: str, skip_first_files: int, load_cap: int) -> Tuple[List[str], int, List[str]]:
        """
        Load text prompts from .txt files in the specified directory.
        
        Args:
            directory_path: Path to the directory containing .txt files
            skip_first_files: Number of files to skip from the beginning (for incremental loading)
            load_cap: Maximum number of prompts to load
            
        Returns:
            Tuple of (list of prompts, total number loaded, list of filenames)
        """
        
        # Validate directory path
        if not directory_path:
            return ([""], 0, [""])
        
        # Expand user path and make absolute
        directory_path = os.path.expanduser(directory_path)
        if not os.path.isabs(directory_path):
            # If relative path, make it relative to ComfyUI base directory
            directory_path = os.path.abspath(directory_path)
        
        if not os.path.exists(directory_path):
            print(f"[LoadPromptsFromDirectory] Directory not found: {directory_path}")
            return ([""], 0, [""])
        
        if not os.path.isdir(directory_path):
            print(f"[LoadPromptsFromDirectory] Path is not a directory: {directory_path}")
            return ([""], 0, [""])
        
        # Find all .txt files in the directory
        txt_files = sorted(glob.glob(os.path.join(directory_path, "*.txt")))
        
        if not txt_files:
            print(f"[LoadPromptsFromDirectory] No .txt files found in: {directory_path}")
            return ([""], 0, [""])
        
        # Apply skip_first_files
        if skip_first_files >= len(txt_files):
            print(f"[LoadPromptsFromDirectory] Skip value ({skip_first_files}) exceeds number of files ({len(txt_files)})")
            return ([""], 0, [""])
        
        txt_files = txt_files[skip_first_files:]
        
        # Apply load_cap
        txt_files = txt_files[:load_cap]
        
        prompts = []
        file_names = []
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    prompts.append(content)
                    file_names.append(os.path.basename(txt_file))
            except Exception as e:
                print(f"[LoadPromptsFromDirectory] Error reading file {txt_file}: {e}")
                prompts.append("")
                file_names.append(os.path.basename(txt_file))
        
        if not prompts:
            return ([""], 0, [""])
        
        total_loaded = len(prompts)
        print(f"[LoadPromptsFromDirectory] Loaded {total_loaded} prompts from {directory_path} (skipped first {skip_first_files} files)")
        
        return (prompts, total_loaded, file_names)


class LoadSinglePromptByIndex:
    """
    Load a single prompt from a directory by its index.
    Useful for iterating through prompts one at a time with an integer counter.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to directory containing .txt files"
                }),
                "file_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("prompt", "file_name", "total_files", "is_valid")
    FUNCTION = "load_single_prompt"
    CATEGORY = "Warper Tools/Prompts"
    
    def load_single_prompt(self, directory_path: str, file_index: int) -> Tuple[str, str, int, bool]:
        """
        Load a single prompt by index from .txt files in the specified directory.
        
        Args:
            directory_path: Path to the directory containing .txt files
            file_index: Index of the file to load (0-based)
            
        Returns:
            Tuple of (prompt text, filename, total number of files, is_valid flag)
        """
        
        # Validate directory path
        if not directory_path:
            return ("", "", 0, False)
        
        # Expand user path and make absolute
        directory_path = os.path.expanduser(directory_path)
        if not os.path.isabs(directory_path):
            directory_path = os.path.abspath(directory_path)
        
        if not os.path.exists(directory_path):
            print(f"[LoadSinglePromptByIndex] Directory not found: {directory_path}")
            return ("", "", 0, False)
        
        if not os.path.isdir(directory_path):
            print(f"[LoadSinglePromptByIndex] Path is not a directory: {directory_path}")
            return ("", "", 0, False)
        
        # Find all .txt files in the directory
        txt_files = sorted(glob.glob(os.path.join(directory_path, "*.txt")))
        
        total_files = len(txt_files)
        
        if not txt_files:
            print(f"[LoadSinglePromptByIndex] No .txt files found in: {directory_path}")
            return ("", "", 0, False)
        
        # Check if index is valid
        if file_index < 0 or file_index >= total_files:
            print(f"[LoadSinglePromptByIndex] Index {file_index} out of range (0-{total_files-1})")
            return ("", "", total_files, False)
        
        txt_file = txt_files[file_index]
        file_name = os.path.basename(txt_file)
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"[LoadSinglePromptByIndex] Loaded prompt from {file_name} (index {file_index})")
                return (content, file_name, total_files, True)
        except Exception as e:
            print(f"[LoadSinglePromptByIndex] Error reading file {txt_file}: {e}")
            return ("", file_name, total_files, False)


class CombinePrompts:
    """
    Combine multiple prompts with customizable separators and formatting.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "prompt1": ("STRING", {"forceInput": True}),
                "prompt2": ("STRING", {"forceInput": True}),
                "prompt3": ("STRING", {"forceInput": True}),
                "prompt4": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {
                    "default": ", ",
                    "multiline": False
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_prompt",)
    FUNCTION = "combine"
    CATEGORY = "Warper Tools/Prompts"
    
    def combine(self, separator=", ", prefix="", suffix="", prompt1=None, prompt2=None, prompt3=None, prompt4=None):
        """
        Combine multiple prompts into a single string.
        """
        prompts = []
        
        # Collect non-empty prompts
        for prompt in [prompt1, prompt2, prompt3, prompt4]:
            if prompt and prompt.strip():
                prompts.append(prompt.strip())
        
        if not prompts:
            return ("",)
        
        # Combine with separator
        combined = separator.join(prompts)
        
        # Add prefix and suffix
        if prefix:
            combined = prefix + combined
        if suffix:
            combined = combined + suffix
        
        return (combined,)
