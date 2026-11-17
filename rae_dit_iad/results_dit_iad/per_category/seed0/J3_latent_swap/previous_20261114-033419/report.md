# RAE-DiT IAD results (J3_latent_swap)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.8344 | 0.8281 | 0.8360 | 0.8985 | 0.8902 | 0.8917 | 0.8427 | 0.2009 | 0.6690 |
| single | 1.0 | 1 | best | latent_cos | 0.8281 | 0.8254 | 0.8475 | 0.8914 | 0.8866 | 0.9021 | 0.9047 | 0.2193 | 0.6504 |
| single | 1.0 | 1 | best | pixel | 0.4055 | 0.5701 | 0.5442 | 0.6352 | 0.7307 | 0.7265 | 0.5757 | 0.0717 | 0.3366 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.5080 | 0.5140 | 0.5562 | 0.8210 | 0.8050 | 0.8383 | 0.3400 | 0.0113 | 0.0660 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9751 | 0.8896 | 0.9358 | 0.9929 | 0.9654 | 0.9810 | 0.9690 | 0.4544 | 0.8896 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9912 | 0.9848 | 0.9908 | 0.9975 | 0.9960 | 0.9976 | 0.9778 | 0.4607 | 0.9156 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4010 | 0.4227 | 0.3901 | 0.6900 | 0.7263 | 0.6875 | 0.5563 | 0.0079 | 0.2178 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 0.9758 | 0.9657 | 0.9808 | 0.9912 | 0.9875 | 0.9933 | 0.9378 | 0.1100 | 0.4648 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.9549 | 0.9382 | 0.9465 | 0.9856 | 0.9799 | 0.9824 | 0.9230 | 0.1427 | 0.7903 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.4704 | 0.8989 | 0.8018 | 0.6406 | 0.9509 | 0.8961 | 0.8003 | 0.2949 | 0.7904 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.7232 | 0.6889 | 0.7336 | 0.8526 | 0.8081 | 0.8452 | 0.8811 | 0.1679 | 0.5968 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.8268 | 0.7786 | 0.8132 | 0.9064 | 0.8847 | 0.8996 | 0.9000 | 0.2158 | 0.6291 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2234 | 0.6309 | 0.5628 | 0.5892 | 0.8339 | 0.7961 | 0.8495 | 0.0103 | 0.5199 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.6864 | 0.8086 | 0.7879 | 0.8577 | 0.9150 | 0.9129 | 0.9316 | 0.1122 | 0.7256 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.5431 | 0.6425 | 0.6276 | 0.7775 | 0.8362 | 0.8238 | 0.7672 | 0.0317 | 0.4981 |
| transistor | single | 1.0 | 1 | transistor_best.pt | pixel | 0.4246 | 0.3837 | 0.4100 | 0.4352 | 0.3375 | 0.4147 | 0.3325 | 0.0341 | 0.0890 |
| transistor | single | 1.0 | 1 | transistor_best.pt | latent_cos | 0.7800 | 0.7742 | 0.7996 | 0.7629 | 0.7572 | 0.7779 | 0.8042 | 0.2524 | 0.5750 |
| transistor | single | 1.0 | 1 | transistor_best.pt | disc | 0.8562 | 0.7967 | 0.8017 | 0.8257 | 0.7544 | 0.7550 | 0.6454 | 0.1537 | 0.5120 |
