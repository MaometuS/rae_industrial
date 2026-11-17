# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.9070 | 0.9800 | 0.9835 | 0.9533 | 0.9905 | 0.9928 | 0.9835 | 0.4353 | 0.9253 |
| single | 0.5 | 1 | best | pixel | 0.4427 | 0.6561 | 0.6278 | 0.7053 | 0.8531 | 0.8336 | 0.7323 | 0.1455 | 0.5217 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.4418 | 0.4747 | 0.5152 | 0.7773 | 0.7827 | 0.8070 | 0.4628 | 0.0164 | 0.1330 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9595 | 0.9976 | 0.9992 | 0.9888 | 0.9993 | 0.9998 | 0.9826 | 0.4155 | 0.9170 |
| grid | single | 0.5 | 1 | grid_best.pt | pixel | 0.4319 | 0.5764 | 0.4887 | 0.7072 | 0.8199 | 0.7524 | 0.7046 | 0.0596 | 0.4281 |
| grid | single | 0.5 | 1 | grid_best.pt | latent_cos | 0.9883 | 0.9992 | 1.0000 | 0.9961 | 0.9997 | 1.0000 | 0.9825 | 0.2046 | 0.9364 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | pixel | 0.5082 | 0.8904 | 0.8525 | 0.6551 | 0.9427 | 0.9199 | 0.8487 | 0.4577 | 0.8316 |
| hazelnut | single | 0.5 | 1 | hazelnut_best.pt | latent_cos | 0.9357 | 0.9779 | 0.9896 | 0.9662 | 0.9835 | 0.9936 | 0.9939 | 0.7820 | 0.9559 |
| screw | single | 0.5 | 1 | screw_best.pt | pixel | 0.3890 | 0.6829 | 0.6548 | 0.6816 | 0.8671 | 0.8551 | 0.9129 | 0.0483 | 0.6939 |
| screw | single | 0.5 | 1 | screw_best.pt | latent_cos | 0.7444 | 0.9453 | 0.9453 | 0.8621 | 0.9794 | 0.9779 | 0.9749 | 0.3390 | 0.8920 |
