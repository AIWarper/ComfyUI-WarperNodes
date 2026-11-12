# ComfyUI Warper Nodes

Lightweight JPEG saving and previews for ComfyUI. This package contains exactly two nodes, both under **Warper Tools/Image**.

## Nodes

| Node | Behavior |
| --- | --- |
| **Save JPEG (Warper)** | Saves every image in a batch directly as `.jpg` in ComfyUI's output directory and displays the saved images in the node. |
| **Preview JPEG (Warper)** | Displays every image in a batch using `.jpg` files in ComfyUI's temporary directory. Does not save to the output directory. |

Connect the `IMAGE` output of **VAE Decode** (or any image-producing node) to either node. No intermediate PNG is created.

Both nodes have a `quality` setting from **1 to 100**, default **95**, and preserve the original resolution. JPEG encoding uses full chroma resolution (4:4:4). Save JPEG also accepts a `filename_prefix`, including subfolders and `%batch_num%`.

JPEG is lossy; file size depends on image content and quality. RGBA inputs lose their alpha channel. These nodes do not embed prompt/workflow metadata; save your workflow separately as JSON. Temporary previews may be removed by ComfyUI. JPEG encoding reduces file sizes, not VAE decoding memory use. Remove or bypass any existing PNG Save/Preview nodes if you no longer want their PNG files.

## Installation

From your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/AIWarper/ComfyUI-WarperNodes.git
```

Restart ComfyUI, then search for **Save JPEG (Warper)** or **Preview JPEG (Warper)**. No extra models, node packs, or dependencies are needed beyond ComfyUI's existing NumPy and Pillow.

To update an existing Git installation, run `git pull` inside `custom_nodes/ComfyUI-WarperNodes` and restart ComfyUI.

## Legacy nodes retired

This is a breaking update: all former pose, mask, optical-flow, batching, resolution/preprocessing, prompt-loading, and text-file iterator nodes have been retired and removed, along with their examples and bundled RAFT/UniMatch code. Only the two JPEG nodes above are registered.

Existing workflows using retired nodes will report missing nodes after updating. If you still need those workflows, keep a separate ComfyUI installation on legacy commit `35e83bb` rather than updating it. The legacy implementation remains available in Git history.

## Development

Run the CPU tests with ComfyUI's Python environment (or Python with NumPy, Pillow, and Torch installed):

```bash
python -m unittest discover -s tests -v
```
