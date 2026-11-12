# RAE-DiT IAD results (J3_latent_swap)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.8439 | 0.8157 | 0.8270 | 0.9111 | 0.8848 | 0.8943 | 0.7624 | 0.1517 | 0.5700 |
| single | 1.0 | 1 | best | latent_cos | 0.8649 | 0.8453 | 0.8698 | 0.9147 | 0.9052 | 0.9213 | 0.9044 | 0.2269 | 0.6386 |
| single | 1.0 | 1 | best | pixel | 0.4104 | 0.5668 | 0.5451 | 0.6359 | 0.7289 | 0.7258 | 0.5767 | 0.0720 | 0.3390 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.4988 | 0.5060 | 0.5554 | 0.8166 | 0.8009 | 0.8383 | 0.3390 | 0.0113 | 0.0680 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 0.9358 | 0.8800 | 0.9298 | 0.9812 | 0.9558 | 0.9755 | 0.9626 | 0.4042 | 0.8612 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.9964 | 0.9924 | 0.9968 | 0.9989 | 0.9976 | 0.9990 | 0.8232 | 0.1395 | 0.7578 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4002 | 0.4211 | 0.3868 | 0.6899 | 0.7262 | 0.6873 | 0.5576 | 0.0080 | 0.2245 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 0.9891 | 0.9724 | 0.9875 | 0.9960 | 0.9898 | 0.9955 | 0.9544 | 0.1277 | 0.4093 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.9181 | 0.9432 | 0.9532 | 0.9697 | 0.9816 | 0.9846 | 0.8180 | 0.0961 | 0.6356 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.4789 | 0.8954 | 0.8000 | 0.6448 | 0.9499 | 0.8959 | 0.8071 | 0.2967 | 0.7908 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.7854 | 0.7196 | 0.7346 | 0.8798 | 0.8160 | 0.8284 | 0.9032 | 0.2236 | 0.6119 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.8868 | 0.8429 | 0.8582 | 0.9415 | 0.9209 | 0.9294 | 0.8924 | 0.3944 | 0.6885 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2517 | 0.6286 | 0.5681 | 0.5982 | 0.8340 | 0.7993 | 0.8503 | 0.0104 | 0.5225 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7489 | 0.7711 | 0.7786 | 0.8866 | 0.9172 | 0.9225 | 0.8938 | 0.0982 | 0.6509 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.5472 | 0.5647 | 0.5624 | 0.7788 | 0.7975 | 0.7918 | 0.6799 | 0.0061 | 0.3701 |
| transistor | single | 1.0 | 1 | transistor_best.pt | pixel | 0.4225 | 0.3829 | 0.4154 | 0.4302 | 0.3336 | 0.4084 | 0.3294 | 0.0337 | 0.0890 |
| transistor | single | 1.0 | 1 | transistor_best.pt | latent_cos | 0.8654 | 0.8833 | 0.9183 | 0.8301 | 0.8474 | 0.8848 | 0.8082 | 0.2806 | 0.6598 |
| transistor | single | 1.0 | 1 | transistor_best.pt | disc | 0.8708 | 0.7354 | 0.7642 | 0.8665 | 0.7264 | 0.7669 | 0.5983 | 0.1224 | 0.3979 |
