# ComfyUI Warper Nodes

A collection of general-purpose nodes for ComfyUI.

## Current Nodes

*   **DWPose Scaler (Warper):** Provides advanced scaling and adjustment options for DWPose keypoints. (Category: `Warper Tools`)
    ![DWPose Scaler Example Image](assets/images/DWScaler.jpg)

*   **Flow Visualizer (Warper):** Calculates optical flow between image sequences using the RAFT model and provides visual representations (color-coded flow images) for both forward and backward flow. Ideal for understanding motion in your generations. (Category: `Warper Tools/Flow`)
    *   **Model Requirement:** This node requires a RAFT model checkpoint.

*   **Smart Video Batcher (Warper):** Splits image sequences into smaller batches with special padding for the final batch. (Category: `Warper Tools/Looping`)

*   **Get Batch By Index (Warper):** Selects a single image batch from a list of batches by its index. (Category: `Warper Tools/Looping`)

*   **Smart Overlapping Batcher (Warper):** Creates overlapping batches from an image sequence to help maintain temporal consistency. (Category: `Warper Tools/Looping`)

*   **Mouth Mask from Pose (Warper):** Generates a circular mask around the mouth area based on facial keypoints. (Category: `Warper Tools`)

*   **Facial Part Mask from Pose (Warper):** Creates masks for various facial parts (entire face, mouth, eyes) using different shapes (convex hull, ellipse, etc.) based on facial keypoints. (Category: `Warper Tools`)

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory.
2.  Clone this repository:
    ```bash
    git clone https://github.com/AIWarper/ComfyUI-WarperNodes.git ComfyUI-WarperNodes
    ```
    *(Note: Added `ComfyUI-WarperNodes` to the clone command to ensure the directory is named consistently, as your `requirements.txt` path below implies this name.)*

3.  **Install RAFT Model (Required for Flow Visualizer Node):**
    *   Download the RAFT model checkpoint: `Tartan-C-T-TSKH-spring540x960-M.pth`
        *   **Download Link:** [Google Drive Folder](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW) (Look for the specified `.pth` file within this folder.)
    *   Create a `raft` folder inside your ComfyUI `models` directory if it doesn't already exist: `ComfyUI/models/raft/`
    *   Place the downloaded `Tartan-C-T-TSKH-spring540x960-M.pth` file into this `ComfyUI/models/raft/` directory.

4.  Install any Python package dependencies (if listed in `requirements.txt`):
    ```bash
    cd ComfyUI-WarperNodes
    pip install -r requirements.txt
    ```
    (If you use ComfyUI's portable version, you might need to use its embedded Python, e.g., `path/to/ComfyUI/python_embeded/python.exe -m pip install -r requirements.txt`. If a `requirements.txt` file is not present, this step may not be needed.)

5.  Restart ComfyUI.

## Usage Notes

*   Ensure the RAFT model is correctly placed as described above for the **Flow Visualizer (Warper)** node to function.
*   The nodes will appear under the "Warper Tools" category in the ComfyUI "Add Node" menu.

---

**TODO:**
*   Add example images/gifs for other nodes.
*   More detailed usage instructions for each node.