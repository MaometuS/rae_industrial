# RAE-DiT IAD results (ab_noise_mask_all)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.5508 | 0.6716 | 0.6649 | 0.6933 | 0.7955 | 0.7926 | 0.4583 | 0.0284 | 0.1506 |
| single | 0.1 | 1 | best | latent_cos | 0.2639 | 0.3693 | 0.3511 | 0.5398 | 0.6070 | 0.5985 | 0.5519 | 0.0282 | 0.3581 |
| single | 0.1 | 1 | best | pixel | 0.4980 | 0.6036 | 0.5917 | 0.6958 | 0.7464 | 0.7483 | 0.7698 | 0.1874 | 0.5712 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.5594 | 0.6120 | 0.6136 | 0.8144 | 0.8620 | 0.8491 | 0.5042 | 0.0507 | 0.2435 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.0758 | 0.2002 | 0.1685 | 0.5704 | 0.6122 | 0.6013 | 0.4319 | 0.0130 | 0.1829 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.6902 | 0.7592 | 0.7689 | 0.8569 | 0.8958 | 0.8980 | 0.1559 | 0.0087 | 0.0178 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.7419 | 0.7836 | 0.8906 | 0.8521 | 0.9019 | 0.9517 | 0.9387 | 0.2193 | 0.8673 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.1546 | 0.3074 | 0.2682 | 0.5609 | 0.6550 | 0.6272 | 0.0455 | 0.0035 | 0.0007 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.6533 | 0.9340 | 0.9098 | 0.8434 | 0.9787 | 0.9713 | 0.2578 | 0.0043 | 0.0776 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.6479 | 0.9329 | 0.9014 | 0.7493 | 0.9660 | 0.9489 | 0.9629 | 0.5630 | 0.8915 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.4054 | 0.3996 | 0.3954 | 0.5806 | 0.5856 | 0.5757 | 0.7609 | 0.0402 | 0.5573 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.6539 | 0.7421 | 0.7329 | 0.7840 | 0.8486 | 0.8447 | 0.6994 | 0.0643 | 0.2718 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.0016 | 0.2080 | 0.1013 | 0.5340 | 0.6155 | 0.5561 | 0.8770 | 0.0283 | 0.5924 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.2257 | 0.4599 | 0.4429 | 0.5925 | 0.7294 | 0.7315 | 0.9034 | 0.0137 | 0.6673 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.3870 | 0.3720 | 0.3726 | 0.6598 | 0.6898 | 0.6890 | 0.5885 | 0.0032 | 0.2030 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.5392 | 0.4817 | 0.4517 | 0.5292 | 0.3865 | 0.4356 | 0.5661 | 0.0757 | 0.2613 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.4583 | 0.4792 | 0.4804 | 0.3945 | 0.4527 | 0.4571 | 0.6177 | 0.0705 | 0.3823 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.3696 | 0.5508 | 0.5404 | 0.3224 | 0.5645 | 0.5601 | 0.5896 | 0.0614 | 0.1830 |
