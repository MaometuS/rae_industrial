# RAE-DiT IAD results (A_flow_matching)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8658 | 0.9614 | 0.9718 | 0.9135 | 0.9687 | 0.9786 | 0.9612 | 0.4311 | 0.8793 |
| single | 0.5 | 1 | joint_best.pt | pixel | 0.4297 | 0.6118 | 0.6065 | 0.6440 | 0.7593 | 0.7617 | 0.6139 | 0.1112 | 0.4099 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.5 | 1 | joint_best.pt | pixel | 0.4964 | 0.5265 | 0.5618 | 0.8209 | 0.8064 | 0.8409 | 0.3647 | 0.0119 | 0.1014 |
| carpet | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.9402 | 0.9900 | 0.9960 | 0.9831 | 0.9969 | 0.9988 | 0.9790 | 0.4860 | 0.9102 |
| grid | single | 0.5 | 1 | joint_best.pt | pixel | 0.4361 | 0.4962 | 0.4737 | 0.7016 | 0.7482 | 0.7352 | 0.6002 | 0.0151 | 0.3145 |
| grid | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.9774 | 0.9967 | 1.0000 | 0.9921 | 0.9988 | 1.0000 | 0.9814 | 0.1930 | 0.9383 |
| hazelnut | single | 0.5 | 1 | joint_best.pt | pixel | 0.4514 | 0.9254 | 0.8586 | 0.6087 | 0.9614 | 0.9220 | 0.8321 | 0.4481 | 0.8240 |
| hazelnut | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8911 | 0.9604 | 0.9796 | 0.9423 | 0.9711 | 0.9882 | 0.9936 | 0.7750 | 0.9563 |
| screw | single | 0.5 | 1 | joint_best.pt | pixel | 0.3150 | 0.6737 | 0.6573 | 0.6427 | 0.8476 | 0.8476 | 0.9019 | 0.0365 | 0.6579 |
| screw | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.6690 | 0.9389 | 0.9375 | 0.8267 | 0.9800 | 0.9782 | 0.9661 | 0.3280 | 0.8771 |
| transistor | single | 0.5 | 1 | joint_best.pt | pixel | 0.4496 | 0.4371 | 0.4813 | 0.4460 | 0.4332 | 0.4627 | 0.3708 | 0.0442 | 0.1517 |
| transistor | single | 0.5 | 1 | joint_best.pt | latent_cos | 0.8512 | 0.9213 | 0.9458 | 0.8234 | 0.8966 | 0.9278 | 0.8859 | 0.3733 | 0.7144 |
