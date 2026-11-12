# RAE-DiT IAD results (ab_noise_sigma02)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.2 | 1 | best | disc | 0.4745 | 0.4881 | 0.4871 | 0.6557 | 0.6580 | 0.6505 | 0.6896 | 0.0471 | 0.2399 |
| single | 0.2 | 1 | best | latent_cos | 0.3769 | 0.4311 | 0.4311 | 0.5911 | 0.6401 | 0.6383 | 0.5698 | 0.0633 | 0.3377 |
| single | 0.2 | 1 | best | pixel | 0.4728 | 0.6416 | 0.6365 | 0.6909 | 0.7881 | 0.7828 | 0.8422 | 0.2437 | 0.6233 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.2 | 1 | carpet_best.pt | pixel | 0.5157 | 0.4924 | 0.4948 | 0.7567 | 0.7649 | 0.7635 | 0.6916 | 0.0801 | 0.2842 |
| carpet | single | 0.2 | 1 | carpet_best.pt | latent_cos | 0.5104 | 0.5361 | 0.5289 | 0.7832 | 0.8223 | 0.8211 | 0.8241 | 0.1015 | 0.6047 |
| carpet | single | 0.2 | 1 | carpet_best.pt | disc | 0.5510 | 0.5100 | 0.5068 | 0.7734 | 0.7593 | 0.7617 | 0.6412 | 0.0223 | 0.1722 |
| grid | single | 0.2 | 1 | grid_best.pt | pixel | 0.7703 | 0.8496 | 0.9323 | 0.8718 | 0.9367 | 0.9746 | 0.9493 | 0.2655 | 0.8785 |
| grid | single | 0.2 | 1 | grid_best.pt | latent_cos | 0.0902 | 0.2464 | 0.2147 | 0.5419 | 0.5988 | 0.5843 | 0.0644 | 0.0035 | 0.0000 |
| grid | single | 0.2 | 1 | grid_best.pt | disc | 0.4578 | 0.7853 | 0.7895 | 0.6823 | 0.9050 | 0.9087 | 0.7650 | 0.0742 | 0.5100 |
| hazelnut | single | 0.2 | 1 | hazelnut_best.pt | pixel | 0.3729 | 0.9500 | 0.9361 | 0.6282 | 0.9762 | 0.9675 | 0.9369 | 0.6146 | 0.8727 |
| hazelnut | single | 0.2 | 1 | hazelnut_best.pt | latent_cos | 0.4639 | 0.4743 | 0.4932 | 0.6017 | 0.6437 | 0.6506 | 0.7846 | 0.1640 | 0.3279 |
| hazelnut | single | 0.2 | 1 | hazelnut_best.pt | disc | 0.2614 | 0.2204 | 0.2250 | 0.5016 | 0.4935 | 0.4911 | 0.7313 | 0.0348 | 0.2102 |
| screw | single | 0.2 | 1 | screw_best.pt | pixel | 0.0004 | 0.2861 | 0.1422 | 0.5340 | 0.6655 | 0.5684 | 0.8651 | 0.0477 | 0.5367 |
| screw | single | 0.2 | 1 | screw_best.pt | latent_cos | 0.5229 | 0.4886 | 0.5079 | 0.7108 | 0.7393 | 0.7430 | 0.9124 | 0.0179 | 0.6977 |
| screw | single | 0.2 | 1 | screw_best.pt | disc | 0.4579 | 0.5030 | 0.4853 | 0.7078 | 0.7469 | 0.7174 | 0.5739 | 0.0026 | 0.0824 |
| transistor | single | 0.2 | 1 | transistor_best.pt | pixel | 0.7046 | 0.6300 | 0.6771 | 0.6638 | 0.5971 | 0.6402 | 0.7679 | 0.2108 | 0.5443 |
| transistor | single | 0.2 | 1 | transistor_best.pt | latent_cos | 0.2971 | 0.4100 | 0.4108 | 0.3179 | 0.3966 | 0.3923 | 0.2633 | 0.0298 | 0.0582 |
| transistor | single | 0.2 | 1 | transistor_best.pt | disc | 0.6446 | 0.4221 | 0.4287 | 0.6137 | 0.3852 | 0.3734 | 0.7366 | 0.1016 | 0.2247 |
