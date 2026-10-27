# Long-tailed skin lesion classification on ISIC 2019

Controlled benchmark of 14 class-imbalance methods and diffusion-based augmentation with matched controls.
Backbone: EfficientNet-B0 at 224 px, 3 seeds, 3-model ensembles with flip test-time augmentation.

Authors: Manthan Laddha, Nisrag Vaghela, Sankhadeep Chatterjee

## Repository layout
| Folder | Contents |
|---|---|
| `notebooks/` | Executed notebooks (outputs included) |
| `src/` | The same code as plain `.py` scripts (`# %%` marks cell boundaries) |
| `results/` | Text outputs of the key cells (tables, logs, paired tests) |
| `figures/` | Figures saved from the notebooks |

## Run order
1. `01_data_and_cache` manifest, lesion-grouped splits, 224/320/512 px cache
2. `02_difficulty_and_methods_benchmark` out-of-fold difficulty scores, 14 methods x 3 seeds, results table
3. `03_generator_512_synthetic_loop` lesion masks, 512 px LoRA inpainting generator, synthetic pool, matched-control ablation, t-SNE, closed loop
4. `04_results_and_paired_tests` combined table and paired bootstrap tests
- `00_legacy_pipeline_256px` earlier pipeline (256 px generator, probe-based dose rule), kept for reference. Its outputs were not saved, and it is not comparable with notebooks 01-04.

## Data
ISIC 2019 training set and official test set with ground truth. Not included; obtain it from the ISIC archive or Kaggle and set `ROOT` in notebook 01.

## Key result files
- `results/final_results_table.txt` all methods, with and without logit adjustment
- `results/Z1_paired_bootstrap.txt` synthetic vs matched controls
- `results/G3_pool_scores_summary.txt` generated-image quality (sharpness ratio, gate pass rate)
- `results/G6_closed_loop.txt` closed-loop dose history

## Known limitations
- Single dataset, single backbone, 224 px classifier, 3 seeds.
- LDAM is a simplified version (no normalised classifier).
- 320 px training and other datasets were not run.
- The prompt A/B and real-vs-synthetic detectability test at 512 px did not complete (`results/G2_gate_attempt.txt` shows the interrupted run).
- Validation sets for DF and VASC are small (49 and 56 images).

## Requirements
Python 3.10+, GPU (T4 class) for training. See `requirements.txt`. Remove `torchao` if PEFT complains about its version (`pip uninstall -y torchao`).
