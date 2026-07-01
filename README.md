# AI Capstone Project with Deep Learning — Agricultural Land Classification

Deep learning pipelines for classifying satellite land tiles as
**agricultural** vs. **non-agricultural**, built with both Keras and
PyTorch, progressing from basic data loading through CNN classifiers to
hybrid CNN–Vision Transformer (CNN-ViT) models.

Built as part of the IBM AI Engineering Professional Certificate
("AI Capstone Project with Deep Learning") on Coursera.

## Contents

| # | Notebook | Description | Result |
|---|---|---|---|
| 1 | `Compare_Memory-Based_Versus_Generator-Based_Data_Loading.ipynb` | Bulk vs. on-demand image loading, timed and memory-profiled | Bulk: 143 MB peak / Generator: ~1 MB peak |
| 2 | `Data_Loading_and_Augmentation_Using_Keras.ipynb` | Custom `keras.utils.Sequence` vs. `image_dataset_from_directory` + `tf.data` | — |
| 3 | `Data_Loading_and_Augmentation_Using_PyTorch.ipynb` | Custom `Dataset` vs. `torchvision.datasets.ImageFolder` | — |
| 4 | `Train_and_Evaluate_a_Keras-Based_Classifier.ipynb` | 3-block CNN (Conv2D/BatchNorm/MaxPool) in Keras | **98.17% val. accuracy** |
| 5 | `Implement_and_Test_a_PyTorch-Based_Classifier.ipynb` | Matching CNN architecture in PyTorch | **96.00% val. accuracy** |
| 6 | `Comparative_Analysis_of_Keras_and_PyTorch_Models.ipynb` | Both CNNs evaluated on an identical split; ROC/AUC | Keras AUC 0.9992 / PyTorch AUC 0.9954 |
| 7 | `Vision_Transformers_in_Keras.ipynb` | Frozen CNN feature extractor → tokens → Transformer encoder blocks | **99.75% val. accuracy** |
| 8 | `Vision_Transformers_in_PyTorch.ipynb` | Same CNN-ViT hybrid design in PyTorch | **100% val. accuracy** |
| 9 | `Land_Classification_CNN-Transformer_Integration_Evaluation.ipynb` | Both hybrid ViT models cross-checked on the same input | 99.83% prediction agreement, AUC 1.0000 (both) |

## Dataset

`images_dataSAT/` — 6,000 real Sentinel-2 satellite tiles (64×64 RGB
JPEG), balanced across two classes:

- `class_0_non_agri/` — 3,000 images
- `class_1_agri/` — 3,000 images

This folder is required (it's not included in this file set — see
"Getting the dataset" below) and must sit in the same directory as the
notebooks. Its internal class-subfolder structure is required by
`ImageFolder` / `image_dataset_from_directory` to infer labels, so this
is the one place a folder structure is unavoidable.

## Trained models

The following checkpoints are produced by notebooks 4, 5, 7, and 8, and
are consumed by notebooks 6 and 9. All model files sit in the repo root
alongside the notebooks (no subfolder):

- `keras_cnn_classifier.keras`
- `pytorch_cnn_classifier.pt`
- `keras_cnn_vit_hybrid.keras`
- `pytorch_cnn_vit_hybrid.pt`

If you re-run notebooks 4/5/7/8 yourself, these files will be
regenerated automatically in the same directory.

## Getting the dataset

`images_dataSAT/` is not part of this file upload (it's ~6,000 image
files, too many to hand-upload individually). To reproduce:

1. If you have the original `images_dataSAT.tar` used to build this
   project, extract it into the repo root: `tar -xf images_dataSAT.tar`
2. Otherwise, use `git` locally to add the folder, since `git add` and
   `git push` handle folder structures natively even though the GitHub
   web upload UI does not:
   ```bash
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

Then open notebooks in order (1 → 9), with `images_dataSAT/` present in
the same directory. Notebooks 6 and 9 require their corresponding
earlier notebooks to have been run first (or use the checkpoints already
provided alongside them).

### Known environment note (CPU-only setups)

If you hit a segmentation fault when TensorFlow and PyTorch are both
imported in the same session (notebooks 6 and 9), it is caused by a
known conflict between PyTorch's bundled Triton JIT compiler and
TensorFlow on some CPU-only environments. Fix:

```bash
pip uninstall triton -y
```

This is safe since these notebooks don't use `torch.compile`.

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
├── pytorch_cnn_vit_hybrid.pt
└── images_dataSAT/            (add separately -- see "Getting the dataset")
    ├── class_0_non_agri/
    └── class_1_agri/
```

## Techniques covered

- Memory-based vs. generator-based data loading, with timing and
  `tracemalloc` memory profiling
- Custom Keras `Sequence` / PyTorch `Dataset` classes vs. framework
  built-ins (`tf.data`, `ImageFolder`)
- Image augmentation (flips, rotation, zoom, contrast)
- CNN architectures with `Conv2D`/`Conv2d`, `BatchNorm`, `MaxPooling`,
  `Dropout`
- Cross-framework model evaluation on an identical held-out split
  (accuracy, precision, recall, F1, ROC/AUC)
- CNN feature-map tokenization, learnable positional embeddings, and
  stacked Transformer encoder blocks (`MultiHeadAttention` /
  `nn.MultiheadAttention`) for a CNN-ViT hybrid architecture
- Cross-framework model loading and inference-format alignment
  (channels-first vs. channels-last)

## Reference

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). *EuroSAT: A
Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover
Classification.* IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing, 12(7), 2217-2226. (Referenced in
notebook 1's discussion of satellite land-use classification.)

## License

Educational project completed as part of the IBM AI Engineering
Professional Certificate on Coursera. Code in this repository may be
reused under the MIT License (see `LICENSE`); course materials and
assignment text remain the property of IBM/Coursera.
