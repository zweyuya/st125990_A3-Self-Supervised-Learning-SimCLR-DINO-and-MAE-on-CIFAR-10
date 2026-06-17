# A3: Self-Supervised Learning — SimCLR, DINO, and MAE on CIFAR-10

This assignment implements and compares three self-supervised representation
learning methods on CIFAR-10: **SimCLR** (contrastive learning), **DINO**
(self-distillation with no labels), and **MAE** (masked autoencoding). No
labels are used during pretraining; labels are only used afterward to train a
linear classifier on top of the frozen learned features (linear evaluation).

## Setup

- **Dataset:** CIFAR-10 (50,000 train / 10,000 test, 32×32 RGB images, 10 classes)
- **Hardware:** NVIDIA GeForce RTX 2080 Ti (11.5 GB VRAM)
- **Framework:** PyTorch 2.6.0+cu124

```bash
pip install torch torchvision timm scikit-learn tqdm matplotlib numpy pillow -q
```

---

## Part 1 — SimCLR

ResNet-18 backbone (3×3 stem, no max-pool, adapted for 32×32 inputs) +
projection head, trained with NT-Xent contrastive loss on two augmented views
of each image.

- Batch size: 256, Epochs: 10
- Final training loss: 5.42 → 4.94
- Average time/epoch: **86.0s**
- **Linear evaluation accuracy: 65.54%**

---

## Part 2 — DINO

ViT-Tiny student/teacher pair (EMA teacher, no gradient), multi-crop
augmentation (2 global crops + *n* local crops), trained with a
self-distillation loss that includes centering and sharpening to prevent
representation collapse — no negative pairs needed.

---

## Part 3 — MAE

Custom ViT encoder (patch size 4, 64 patches per image) that only processes
*visible* (unmasked) patches, paired with a shallow decoder (4 layers,
128-dim) that reconstructs masked pixel patches. Loss is mean-squared error
on masked patches only, with `norm_pix_loss` normalization.

---

## Exercise 1 — DINO Ablations

Three DINO variants were trained from scratch (10 epochs, batch size 64) to
study the role of centering and local crops.

| Setting | Linear Eval Accuracy |
|---|---|
| Default (2 global + 4 local, with centering) | 50.65% |
| No centering (`- self.center` removed) | 29.12% |
| No local crops (`n_local=0`) | 45.66% |

**Average training time per epoch:**

| Setting | Time/epoch |
|---|---|
| Default | 318.2s |
| No centering | 158.3s |
| No local crops | 66.9s |

### 1a — Center norm across training epochs

`dino_loss_fn.center.norm()` was logged every epoch for the default
(centering-enabled) run and the no-centering run, and plotted across the 10
training epochs. In the default run, the center norm grows during early
training and then stabilizes/plateaus as the EMA-updated center converges to
track the mean teacher output distribution. In the no-centering run, the
center value is still tracked for comparison (it has no effect on the loss in
that variant), allowing a side-by-side view of how the teacher's output
statistics diverge once the corrective centering term is removed from
training.

### 1b — Why does removing centering cause collapse, and why does removing local crops hurt representation quality?

Removing centering caused the most severe degradation, dropping linear
evaluation accuracy from 50.65% to 29.12% — a 21.5 percentage point collapse,
bringing performance much closer to the 10% chance baseline for 10-class
classification than to the default result. This happens because DINO has no
negative pairs and no contrastive term in its loss; the only training signal
is the student matching the teacher's output distribution. Centering
subtracts a running average of the teacher's logits before the softmax,
which prevents the teacher from drifting toward a constant, input-independent
output. Without it, there is nothing constraining the teacher to produce
different outputs for different inputs, so the teacher (and consequently the
student, since it is trained to imitate the teacher) can settle into a
degenerate solution that minimizes the self-distillation loss while encoding
little to no information about the actual image content. The sharp accuracy
drop observed here is direct evidence that this partial collapse occurred.

Removing local crops caused a smaller but still meaningful drop, from 50.65%
to 45.66% (5 percentage points). This is a much milder failure mode than
centering removal because the global-to-global comparison alone is still a
valid, non-trivial training signal — two large, overlapping crops of the same
image still require the network to learn some real invariances to recognize
them as the same. However, DINO's local crops serve a specific purpose beyond
just adding more views: they force the student to predict the teacher's
global, semantically stable output from small, zoomed-in regions that may
show only part of an object. This local-to-global correspondence task pushes
the network to associate fine-grained visual details with whole-object-level
semantics, which is exactly the kind of multi-scale understanding that
produces strong transferable features. Without local crops, the model only
ever learns from global-to-global comparisons, losing this multi-scale
signal and producing weaker, less semantically rich representations, even
though it remains far more functional than the collapsed no-centering
variant.

---

## Exercise 2 — MAE Mask Ratio Sweep

Three MAE models were trained from scratch for 5 epochs each at different
masking ratios.

| Mask Ratio | Recon Loss | Linear Eval Acc |
|---|---|---|
| 0.25 | 0.3664 | 37.42% |
| 0.50 | 0.4554 | 37.56% |
| 0.75 | 0.5903 | 40.10% |

**Average training time per epoch:**

| Mask Ratio | Time/epoch |
|---|---|
| 0.25 | 22.9s |
| 0.50 | 23.9s |
| 0.75 | 23.6s |

### Why does very low masking (e.g. 0.25) produce worse representations even though reconstruction loss is lower?

At `mask_ratio=0.25`, only 25% of patches are hidden, so the encoder has
access to the vast majority of the image when forming its representation.
Reconstructing the small number of masked patches becomes an easy task, since
the model can largely rely on local interpolation: each masked patch is
almost always surrounded by several visible, unmasked neighbors, and natural
images have strong local spatial correlation (color, texture, and edges tend
to vary smoothly between adjacent patches). This lets the model fill in
masked regions using low-level continuity cues rather than developing any
deep understanding of the object or scene as a whole, which is why the
reconstruction loss is lowest here (0.3664) — the task is simply easier, not
because the representation learned is better.

At `mask_ratio=0.75`, three-quarters of the patches are removed, which means
large, contiguous regions of the image are missing and cannot be filled in
using nearby visible patches alone, since there often are no nearby visible
patches. To reconstruct these large gaps, the encoder is forced to infer
high-level structure: what object is likely present, what its overall shape
and parts look like, and how the visible fragments relate to the whole. This
pushes the encoder to build a genuinely semantic, holistic representation of
the image rather than a shallow, locally-interpolated one. This explains why
reconstruction loss is highest at 0.75 (0.5903, since the task is
intrinsically harder) while linear evaluation accuracy is also highest
(40.10%), because the representation learned in the process of solving this
harder task transfers better to downstream classification.

This produces an inverse relationship between reconstruction loss and
representation quality across the sweep: lower loss at 0.25 reflects an
easier but less informative pretraining task, while higher loss at 0.75
reflects a harder task that forces more useful, semantic features into the
encoder. The results confirm this directly — accuracy increases monotonically
with mask ratio even as loss also increases, showing that reconstruction loss
alone is not a reliable proxy for representation quality in masked
autoencoding.

---

## Exercise 3 — Three-Way Comparison (SimCLR vs DINO vs MAE)

| Metric | SimCLR | DINO | MAE |
|---|---|---|---|
| Backbone | ResNet-18 | ViT-Tiny | ViT |
| Needs negative pairs? | Yes | No | No |
| Needs EMA teacher? | No | Yes | No |
| Linear Eval Accuracy | 65.54% | 50.65% | 40.10% |
| Training time/epoch | 86.0s | 318.2s | 23.6s |
| t-SNE cluster quality (1-5) | 4 | 2 | 2 |
| Has interpretable attention maps? | No | Yes | No |

t-SNE cluster quality was assessed visually: SimCLR's embeddings show clear
partial clustering, with airplane, truck, automobile, and ship forming
visibly distinct regions, while the visually similar classes (cats, dogs,
deer, birds) remain mixed in the center. DINO's and MAE's embeddings show
almost no visible clustering, with colors largely intermixed throughout the
plot and only marginal structure at the edges, consistent with their lower
linear evaluation accuracies.

### 3a — Why MAE won out over DINO for large-scale general pre-training, and why DINO is still preferred for CV-only tasks like segmentation

**Two reasons MAE won out over DINO for large-scale general pre-training:**

1. *Compute efficiency.* MAE's encoder only processes the visible patches
   (e.g. 25% at `mask_ratio=0.75`), making each training step far cheaper than
   DINO, which runs the full student network over every global **and** local
   crop, plus a separate teacher forward pass, every step. This is reflected
   directly in the results above: MAE trained at 23.6s/epoch versus DINO's
   318.2s/epoch — roughly 13x faster per epoch for MAE.
2. *Simplicity and stability at scale.* MAE has no EMA teacher, no centering
   mechanism, and no collapse failure mode to guard against (as demonstrated
   in Exercise 1, where removing centering caused DINO's accuracy to drop by
   21.5 points). MAE's reconstruction objective is straightforward, making it
   easier to scale to huge datasets and models without the delicate
   hyperparameter tuning DINO requires (teacher temperature, EMA momentum,
   centering momentum, multi-crop schedule).

**One reason DINO is still preferred for CV-only tasks like segmentation:**

DINO's attention maps are directly interpretable and tend to localize
foreground objects without any segmentation supervision, because the
self-distillation objective on global semantic crops encourages the `[CLS]`
token to attend to object-level structure. MAE's patch-reconstruction
objective optimizes for pixel-level fidelity, which does not push the same
kind of object-centric, segmentation-friendly attention pattern — consistent
with the comparison table, where only DINO is marked as having interpretable
attention maps.

### 3b — Medical image segmentation with 500 labeled scans: which pre-training approach, and why?

With only 500 labeled scans, the downstream task is both label-scarce and
inherently spatial/dense — segmentation requires per-pixel decisions, not
just a single global class label — which favors a pretraining method whose
learned signal already encodes spatial/object structure that transfers well
to pixel-dense tasks. DINO is the stronger choice here, precisely for the
reason given in 3a: its attention maps already localize semantically
meaningful regions without any supervision, which is a much closer match to
what a segmentation head needs to learn from a small labeled set. MAE's
representations, by contrast, are tuned more for global classification-style
linear probes — reflected in this assignment's own results, where MAE scored
the lowest linear evaluation accuracy (40.10%) of the three methods — and
would likely require more labeled data to adapt its features to dense,
pixel-level prediction.

---

## Summary

| Method | Backbone | Linear Eval Accuracy | Time/epoch |
|---|---|---|---|
| SimCLR | ResNet-18 | 65.54% | 86.0s |
| DINO (default) | ViT-Tiny | 50.65% | 318.2s |
| MAE (mask=0.75) | ViT | 40.10% | 23.6s |

SimCLR achieved the strongest linear evaluation accuracy and the clearest
t-SNE clustering of the three methods on this CIFAR-10 setup, though DINO and
MAE each offer distinct advantages explored above — DINO for interpretable,
object-centric attention useful in dense prediction tasks, and MAE for
training efficiency at scale.

For Model Results: https://drive.google.com/drive/folders/1vfEF8qMEUhqpIVzRIit1Z06zNklJXWeR?usp=sharing
