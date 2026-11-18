# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | best | latent_cos | 0.7520 | 0.9421 | 0.9518 | 0.8787 | 0.9808 | 0.9836 | 0.9724 | 0.3988 | 0.8841 |
| single | 0.5 | 1 | best | pixel | 0.4381 | 0.6269 | 0.6308 | 0.7454 | 0.8388 | 0.8492 | 0.6607 | 0.0248 | 0.3906 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | carpet_best.pt | pixel | 0.5305 | 0.5518 | 0.5791 | 0.8362 | 0.8208 | 0.8453 | 0.4245 | 0.0141 | 0.1221 |
| carpet | single | 0.5 | 1 | carpet_best.pt | latent_cos | 0.9049 | 0.9699 | 0.9835 | 0.9722 | 0.9908 | 0.9950 | 0.9767 | 0.5127 | 0.9012 |
| screw | single | 0.5 | 1 | screw_best.pt | pixel | 0.3458 | 0.7020 | 0.6825 | 0.6545 | 0.8569 | 0.8531 | 0.8969 | 0.0354 | 0.6591 |
| screw | single | 0.5 | 1 | screw_best.pt | latent_cos | 0.5991 | 0.9143 | 0.9201 | 0.7851 | 0.9709 | 0.9721 | 0.9681 | 0.2850 | 0.8669 |
