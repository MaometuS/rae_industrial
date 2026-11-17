# RAE-DiT IAD results (J3_latent_swap)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.8389 | 0.8174 | 0.8297 | 0.8985 | 0.8827 | 0.8907 | 0.8302 | 0.1919 | 0.6592 |
| single | 1.0 | 1 | best | latent_cos | 0.8239 | 0.8200 | 0.8508 | 0.8826 | 0.8848 | 0.9036 | 0.8974 | 0.2040 | 0.6350 |
| single | 1.0 | 1 | best | pixel | 0.4086 | 0.5704 | 0.5451 | 0.6366 | 0.7306 | 0.7267 | 0.5761 | 0.0716 | 0.3369 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.5157 | 0.5165 | 0.5606 | 0.8261 | 0.8063 | 0.8406 | 0.3440 | 0.0114 | 0.0682 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9675 | 0.8724 | 0.9217 | 0.9902 | 0.9540 | 0.9753 | 0.9727 | 0.4510 | 0.9055 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9912 | 0.9900 | 0.9920 | 0.9974 | 0.9972 | 0.9978 | 0.9776 | 0.4765 | 0.9091 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.3993 | 0.4244 | 0.3885 | 0.6891 | 0.7265 | 0.6870 | 0.5512 | 0.0077 | 0.2154 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 0.9875 | 0.9641 | 0.9833 | 0.9955 | 0.9872 | 0.9941 | 0.9389 | 0.1116 | 0.5225 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.9691 | 0.9524 | 0.9724 | 0.9902 | 0.9847 | 0.9911 | 0.9222 | 0.1414 | 0.7985 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.4686 | 0.8986 | 0.8004 | 0.6400 | 0.9506 | 0.8957 | 0.7999 | 0.2945 | 0.7900 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.7243 | 0.7229 | 0.7232 | 0.8414 | 0.8242 | 0.8242 | 0.8379 | 0.1124 | 0.4685 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.8014 | 0.7689 | 0.7993 | 0.8894 | 0.8795 | 0.8930 | 0.8795 | 0.1868 | 0.6390 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2304 | 0.6313 | 0.5647 | 0.5912 | 0.8352 | 0.7975 | 0.8500 | 0.0103 | 0.5217 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.6772 | 0.7553 | 0.7950 | 0.8484 | 0.9020 | 0.9169 | 0.9321 | 0.0977 | 0.7292 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.5759 | 0.6155 | 0.6180 | 0.7893 | 0.8289 | 0.8201 | 0.7595 | 0.0276 | 0.4924 |
| transistor | single | 1.0 | 1 | transistor_best.pt | pixel | 0.4292 | 0.3812 | 0.4113 | 0.4368 | 0.3344 | 0.4127 | 0.3352 | 0.0342 | 0.0890 |
| transistor | single | 1.0 | 1 | transistor_best.pt | latent_cos | 0.7629 | 0.7854 | 0.8308 | 0.7374 | 0.7567 | 0.8073 | 0.8056 | 0.2472 | 0.5494 |
| transistor | single | 1.0 | 1 | transistor_best.pt | disc | 0.8567 | 0.7600 | 0.7671 | 0.8260 | 0.7232 | 0.7512 | 0.6124 | 0.1274 | 0.4569 |
