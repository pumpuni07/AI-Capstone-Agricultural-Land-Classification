# AI Capstone Project with Deep Learning — Agricultural Land Classification

Deep learning pipelines for classifying satellite land tiles as
**agricultural** vs. **non-agricultural**, built with both Keras and
PyTorch, progressing from basic data loading through CNN classifiers to
hybrid CNN–Vision Transformer (CNN-ViT) models. Built to match the exact
task-by-task grading rubric of the IBM AI Engineering Professional
Certificate's "AI Capstone Project with Deep Learning."

## Contents

| # | Notebook | Result |
|---|---|---|
| 1 | `Compare_Memory-Based_Versus_Generator-Based_Data_Loading.ipynb` | Bulk: 143 MB peak / Generator: ~1 MB peak |
| 2 | `Data_Loading_and_Augmentation_Using_Keras.ipynb` | `custom_data_generator`, batch size 8 |
| 3 | `Data_Loading_and_Augmentation_Using_PyTorch.ipynb` | `custom_transform`, `imagefolder_loader` |
| 4 | `Train_and_Evaluate_a_Keras-Based_Classifier.ipynb` | 4 Conv2D + 5 Dense, **98.00% val. accuracy** |
| 5 | `Implement_and_Test_a_PyTorch-Based_Classifier.ipynb` | **99.50% val. accuracy** |
| 6 | `Comparative_Analysis_of_Keras_and_PyTorch_Models.ipynb` | `print_metrics`, false-negative analysis |
| 7 | `Vision_Transformers_in_Keras.ipynb` | `build_cnn_vit_hybrid`, **98.83% val. accuracy** |
| 8 | `Vision_Transformers_in_PyTorch.ipynb` | Full spec (768-dim/12-head/12-block) — see compute note below |
| 9 | `Land_Classification_CNN-Transformer_Integration_Evaluation.ipynb` | `KerasViT` 99.33% acc. vs. `PyTorchViT` (undertrained, see note) |

## Important: Lab 8 compute note

Lab 8's rubric specifies a full-scale transformer (embed_dim=768, 12
attention heads, 12 transformer blocks — roughly BERT-base sized, ~85M
parameters). This was built and genuinely trained to the exact
specification, but a CPU-only sandbox environment could only complete a
small fraction of the required training within the available time —
benchmarked at ~2.3s/training step, meaning a full run would take
several hours of CPU compute. The notebook honestly reports exactly how
many batches were completed rather than faking full training; as a
result, that specific model is genuinely undertrained (near-random
performance: ~56% accuracy), which is also reflected honestly in Lab
9's evaluation. **To get meaningful accuracy from this configuration,
re-run Lab 8 in an environment with a GPU** (e.g. Google Colab) or allow
several hours of uninterrupted CPU time, then re-run Lab 9 to reflect
the improved model.

## Dataset

`images_dataSAT/` — 6,000 real Sentinel-2 satellite tiles (64×64 RGB
JPEG): `class_0_non_agri/` (3,000) and `class_1_agri/` (3,000). Not
included in this file set (too many files to hand-upload) — see "Getting
the dataset" below.

## Trained model / weight files

Produced by notebooks 4, 5, 7, and 8; consumed by notebooks 6, 7, and 9.
All sit flat in the repo root:

- `keras_cnn_classifier.keras`
- `pytorch_cnn_classifier.pt`
- `keras_cnn_vit_hybrid.keras` and `keras_cnn_vit_hybrid.weights.h5`
  (the `.weights.h5` file is what Lab 9 actually loads — see note below)

**Not included: `pytorch_cnn_vit_hybrid.pt`** (the full-spec PyTorch ViT
hybrid from Lab 8). At ~341 MB, it exceeds what's practical to hand-upload
and, given the compute note above, that checkpoint reflects an
undertrained model anyway — not a strong artifact worth the repo size.
Re-run Lab 8 to regenerate it locally (it will save automatically to
`pytorch_cnn_vit_hybrid.pt`); if you do want to version a large model
file like this, use [Git LFS](https://git-lfs.com/):

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes pytorch_cnn_vit_hybrid.pt
git commit -m "Add PyTorch ViT hybrid checkpoint via Git LFS"
git push
```

### Why both a `.keras` file and a `.weights.h5` file for the Keras ViT hybrid?

Keras 3 has a known limitation deserializing custom layers that wrap a
nested sub-model (our CNN feature extractor wrapped inside the ViT
hybrid). The full `.keras` file is saved for completeness/inspection,
but Lab 9 robustly loads the model by **rebuilding the identical
architecture in code and calling `load_weights()`** on the
`.weights.h5` file instead — the same pattern already used for the
PyTorch models via `state_dict`.

## Getting the dataset

```bash
# If you have the original tar used to build this project:
tar -xf images_dataSAT.tar

# Otherwise, add your own copy locally via git (git handles folders
# natively even though GitHub's web upload UI does not):
git add images_dataSAT/
git commit -m "Add dataset"
git push
```

## Setup

```bash
git clone <this-repo-url>
cd <repo-name>
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open notebooks in order (1 → 9), with `images_dataSAT/` present in the
same directory. Notebooks 6, 7, and 9 require their corresponding
earlier notebooks to have been run first (or use the checkpoints already
provided). Notebook 9 additionally requires Lab 8 to have been re-run
locally to regenerate `pytorch_cnn_vit_hybrid.pt` (see note above).

### Known environment note (CPU-only setups)

If you hit a segmentation fault when TensorFlow and PyTorch are both
imported in the same session (notebooks 6 and 9), it's a known conflict
between PyTorch's bundled Triton JIT compiler and TensorFlow on some
CPU-only environments:

```bash
pip uninstall triton -y
```

Safe, since these notebooks don't use `torch.compile`.

## Repository layout

All files sit flat in the repo root — no wrapper folders — except the
dataset itself, which needs its class-subfolder structure.

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── Compare_Memory-Based_Versus_Generator-Based_Data_Loading.ipynb
├── Data_Loading_and_Augmentation_Using_Keras.ipynb
├── Data_Loading_and_Augmentation_Using_PyTorch.ipynb
├── Train_and_Evaluate_a_Keras-Based_Classifier.ipynb
├── Implement_and_Test_a_PyTorch-Based_Classifier.ipynb
├── Comparative_Analysis_of_Keras_and_PyTorch_Models.ipynb
├── Vision_Transformers_in_Keras.ipynb
├── Vision_Transformers_in_PyTorch.ipynb
├── Land_Classification_CNN-Transformer_Integration_Evaluation.ipynb
├── keras_cnn_classifier.keras
├── pytorch_cnn_classifier.pt
├── keras_cnn_vit_hybrid.keras
├── keras_cnn_vit_hybrid.weights.h5
├── pytorch_cnn_vit_hybrid.pt   (regenerate locally -- see note above)
└── images_dataSAT/             (add separately -- see "Getting the dataset")
    ├── class_0_non_agri/
    └── class_1_agri/
```

## License

Educational project completed as part of the IBM AI Engineering
Professional Certificate on Coursera. Code in this repository may be
reused under the MIT License (see `LICENSE`); course materials and
assignment text remain the property of IBM/Coursera.
