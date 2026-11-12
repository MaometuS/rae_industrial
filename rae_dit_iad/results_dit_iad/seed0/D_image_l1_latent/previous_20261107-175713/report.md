# RAE-DiT IAD results (D_image_l1_latent)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5647 | 0.4872 | 0.4953 | 0.7291 | 0.6642 | 0.6808 | 0.5059 | 0.0200 | 0.0889 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4392 | 0.3936 | 0.3991 | 0.6293 | 0.6196 | 0.6162 | 0.4863 | 0.0219 | 0.1883 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4623 | 0.5965 | 0.5823 | 0.6674 | 0.7415 | 0.7409 | 0.6065 | 0.0961 | 0.3554 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.4659 | 0.5405 | 0.5357 | 0.7682 | 0.8123 | 0.8087 | 0.3388 | 0.0114 | 0.0671 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2781 | 0.3222 | 0.2689 | 0.6755 | 0.7225 | 0.6775 | 0.6363 | 0.0269 | 0.2172 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.5433 | 0.4398 | 0.4635 | 0.8190 | 0.7442 | 0.7591 | 0.6720 | 0.0222 | 0.0923 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.3977 | 0.4670 | 0.4319 | 0.6909 | 0.7411 | 0.7045 | 0.5971 | 0.0089 | 0.2663 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7719 | 0.3400 | 0.4311 | 0.8668 | 0.6615 | 0.7137 | 0.6687 | 0.0128 | 0.2981 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.4770 | 0.4987 | 0.4987 | 0.7174 | 0.7292 | 0.7250 | 0.4414 | 0.0056 | 0.0901 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.5250 | 0.9139 | 0.8543 | 0.6819 | 0.9583 | 0.9232 | 0.9046 | 0.4137 | 0.8586 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3804 | 0.4625 | 0.4496 | 0.5587 | 0.6190 | 0.6085 | 0.4077 | 0.0163 | 0.1848 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7918 | 0.4036 | 0.3771 | 0.8688 | 0.5917 | 0.5839 | 0.6052 | 0.0245 | 0.0168 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.4925 | 0.6874 | 0.6948 | 0.7641 | 0.8669 | 0.8736 | 0.8454 | 0.0122 | 0.5159 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4501 | 0.5017 | 0.4815 | 0.7358 | 0.7765 | 0.7595 | 0.1901 | 0.0014 | 0.0043 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.4946 | 0.5288 | 0.5290 | 0.7678 | 0.7773 | 0.7833 | 0.3314 | 0.0017 | 0.0560 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.4304 | 0.3738 | 0.3950 | 0.4319 | 0.3291 | 0.3944 | 0.3465 | 0.0344 | 0.0692 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3154 | 0.3417 | 0.3646 | 0.3097 | 0.3184 | 0.3219 | 0.5286 | 0.0523 | 0.2373 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5167 | 0.5650 | 0.6083 | 0.4724 | 0.4787 | 0.5525 | 0.4796 | 0.0457 | 0.1895 |
