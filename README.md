# Text-Conditioned Latent Diffusion for Educational Diagrams

This project builds a text-conditioned image generation pipeline for educational diagrams by:
- Merging AI2D and ScienceQA image–caption data
- Computing CLIP text embeddings
- Fine-tuning the Stable Diffusion VAE on the merged dataset
- Training a lightweight text-conditioned UNet with a DDIM sampler on VAE latents
- Running inference and evaluating with PSNR, SSIM, and a CLIP-based similarity score

The repository currently uses Jupyter notebooks organized by task. No folder restructuring is required.

## Current layout (as-is)
- **Merging_AI2D_ScienceQA/**
  - `Merging_ai2d&scienceqa_Code.ipynb` — builds the unified dataset, computes CLIP embeddings, and can save `educational_diagram_data.pt`
  - `ai2d_labeled.json`, `scienceqa_labeled.json`, `combined_dataset.jsonl`
- **VAE/**
  - `Final_Fine-tuned_VAE_Implementation.ipynb` — fine-tunes the VAE; evaluates; extracts latents
  - `best_fine_tuned_vae.pth` — fine-tuned VAE weights (large)
  - `latent_dataset/latent_dataset/latent_text_data.pt` — latents and text embeddings (note nested folder)
- **LDM/**
  - `Final_LDM Implementation.ipynb` — trains a text-conditioned UNet with cross-attention on 512-d CLIP embeddings, DDIM schedule, EMA
- **Inference/**
  - `Inference_LDM.ipynb` — loads the fine-tuned VAE and trained UNet(+EMA), runs denoising, saves images, and computes metrics
- **Results/**
  - `inference_results/` — generated images and progress grids
- Other: `Base_Architecture.png`, `DATA255_Group08_Project_Report.pdf`

## Environment and prerequisites
- **Python**: 3.10+ recommended
- **GPU**: CUDA GPU highly recommended for training and inference
- **Hugging Face access**: The notebooks load the Stable Diffusion v1.5 VAE via
  `runwayml/stable-diffusion-v1-5` (subfolder `vae`). You may need to accept the model terms on Hugging Face and be logged in.

### Python packages (install these)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.33.1 transformers==4.51.3 accelerate==1.6.0
pip install torchmetrics[image]==1.7.1 torch-fidelity==0.3.0
pip install scikit-image scipy matplotlib pillow tqdm numpy huggingface-hub safetensors
```
Adjust the PyTorch CUDA wheel URL to your system (or use CPU wheels if needed).

## Data and model overview
- **Text embeddings**: CLIP base `openai/clip-vit-base-patch32` (hidden size 512). The UNet cross-attention dimension is set to 512 to match.
- **Images**: resized to `256×256`, normalized to `[0,1]` when stored, `[-1,1]` where required by VAE.
- **VAE**: Stable Diffusion v1.5 VAE, fine-tuned on the merged dataset; scaling factor `≈0.18215` is used throughout.
- **Latents**: VAE latents in shape `[N, 4, 32, 32]` are used to train the diffusion model.
- **Sampler**: Custom DDIM scheduler; EMA maintained for the UNet.

## End-to-end workflow (using existing notebooks)
- **1) Build the merged dataset (captions + CLIP text embeddings)**
  - Open `Merging_AI2D_ScienceQA/Merging_ai2d&scienceqa_Code.ipynb`.
  - It creates `combined_dataset.jsonl` and can save `educational_diagram_data.pt` containing:
    - `images` `[N, 3, 256, 256]`
    - `text_embeddings` `[N, 512]`
    - `labels` `[N]` indicating domain (AI2D/ScienceQA)
  - If you already have `combined_dataset.jsonl`, you can skip rebuilding it.

- **2) Fine-tune the VAE**
  - Open `VAE/Final_Fine-tuned_VAE_Implementation.ipynb`.
  - Make sure it loads your `educational_diagram_data.pt` (path tips below).
  - Train until early stop; the notebook saves `VAE/best_fine_tuned_vae.pth`.
  - It also includes evaluation (MSE/SSIM) and visualization cells.

- **3) Extract VAE latents**
  - In the same VAE notebook (section that extracts latents), run the cell to export latents.
  - Output file (as saved by the notebook):
    - `VAE/latent_dataset/latent_dataset/latent_text_data.pt`
    - Contains keys: `z` `[N, 4, 32, 32]` and `text_embeddings` `[N, 512]`.

- **4) Train the text-conditioned UNet (LDM)**
  - Open `LDM/Final_LDM Implementation.ipynb`.
  - Ensure it loads the latents and (optionally) the fine-tuned VAE scaling factor.
  - The notebook trains the UNet with cross-attention over the CLIP embeddings and maintains EMA; best weights are saved under `outputs/` in the notebook’s working directory.

- **5) Inference and evaluation**
  - Open `Inference/Inference_LDM.ipynb`.
  - It loads: the fine-tuned VAE, the UNet (EMA if available), and `latent_text_data.pt`.
  - Produces progression images and final generations; computes PSNR/SSIM and CLIP-based similarity.
  - Saves to `Inference/inference_results/` (or `./inference_results/` relative to the notebook).

## Important path notes (run-as-is with current structure)
The notebooks use simple relative paths. With the current folder layout, you have two options: either copy the referenced files into each notebook’s working directory, or change the string paths in a few places. Here are the common path fixes:

- **Where the files currently are**
  - Fine-tuned VAE: `VAE/best_fine_tuned_vae.pth`
  - Latents: `VAE/latent_dataset/latent_dataset/latent_text_data.pt`
  - LDM best EMA weights (after training): `LDM/outputs/best_diffusion_model_ema.pth` (created by the LDM notebook)

- **LDM notebook (training)**
  - If it does `data = torch.load("latent_text_data.pt")`, change to:
    ```python
    data = torch.load("../VAE/latent_dataset/latent_dataset/latent_text_data.pt")
    ```

- **Inference notebook**
  - VAE weights load:
    ```python
    ckpt = torch.load("../VAE/best_fine_tuned_vae.pth", map_location=device)
    ```
  - Latents load:
    ```python
    data = torch.load("../VAE/latent_dataset/latent_dataset/latent_text_data.pt")
    ```
  - EMA UNet weights (after training with the LDM notebook):
    ```python
    ema_state = torch.load("../LDM/outputs/best_diffusion_model_ema.pth", map_location=device)
    ```
  - If you trained without EMA, update the non-EMA path similarly.

- **VAE notebook**
  - If it does `data = torch.load("educational_diagram_data.pt")` but the file is in the merging folder, use:
    ```python
    data = torch.load("../Merging_AI2D_ScienceQA/educational_diagram_data.pt")
    ```
  - Or run the saving cell in the merging notebook again to write the `.pt` file where you prefer to keep it.

## How to run (quick start)
- **Setup**
  - Install the packages listed above and ensure access to the SD v1.5 VAE on Hugging Face.
- **Data prep**
  - Run `Merging_ai2d&scienceqa_Code.ipynb` to build `combined_dataset.jsonl` and `educational_diagram_data.pt`.
- **VAE**
  - Run `Final_Fine-tuned_VAE_Implementation.ipynb` to train and save `best_fine_tuned_vae.pth`.
  - Run the latent extraction cell to generate `latent_text_data.pt`.
- **LDM**
  - Run `Final_LDM Implementation.ipynb` to train the UNet; confirm `outputs/best_diffusion_model_ema.pth` is saved.
- **Inference/Eval**
  - Run `Inference_LDM.ipynb` to generate images and compute metrics; see `inference_results/`.

## Metrics
- **PSNR** and **SSIM** are computed via `scikit-image` (or `torchmetrics` for SSIM in some cells).
- **CLIP-based similarity** uses CLIP image features vs. provided text embedding (cosine similarity scaled to [0,100]).

## Troubleshooting
- **Model access errors**: Accept the license for `runwayml/stable-diffusion-v1-5` on Hugging Face and login with a token if requested.
- **CUDA OOM**: Reduce batch size, number of steps, or switch to CPU for quick tests (slower).
- **File not found**: See the path notes above; update the string paths or copy files next to the notebook.
- **Embedding shape mismatch**: Ensure text embeddings are 512-d from `openai/clip-vit-base-patch32`; reshape to `[B, 1, 512]` where required by the UNet forward.

## Acknowledgments
- Stable Diffusion VAE from `runwayml/stable-diffusion-v1-5` (Hugging Face)
- CLIP `openai/clip-vit-base-patch32`

---
If you want this README updated with exact command blocks for your local paths (or converted to scripts instead of notebooks), let me know and I can adjust it accordingly.
