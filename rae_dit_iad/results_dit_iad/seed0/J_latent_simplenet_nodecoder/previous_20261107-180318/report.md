# RAE-DiT IAD results (J_latent_simplenet_nodecoder)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5530 | 0.4876 | 0.4909 | 0.7216 | 0.6639 | 0.6760 | 0.5068 | 0.0199 | 0.0920 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4670 | 0.4100 | 0.4352 | 0.6511 | 0.6244 | 0.6391 | 0.6449 | 0.0381 | 0.3472 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4086 | 0.5713 | 0.5442 | 0.6357 | 0.7293 | 0.7220 | 0.5711 | 0.0742 | 0.3277 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5152 | 0.5273 | 0.5586 | 0.8246 | 0.8150 | 0.8387 | 0.3543 | 0.0117 | 0.0724 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5670 | 0.3744 | 0.4025 | 0.8349 | 0.7180 | 0.7344 | 0.8452 | 0.0822 | 0.5207 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.4876 | 0.4406 | 0.4691 | 0.7928 | 0.7443 | 0.7609 | 0.6714 | 0.0223 | 0.0995 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.3968 | 0.4403 | 0.4060 | 0.6899 | 0.7231 | 0.6937 | 0.5272 | 0.0072 | 0.2053 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7995 | 0.4127 | 0.5505 | 0.9024 | 0.6969 | 0.7656 | 0.7730 | 0.0197 | 0.4718 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.4770 | 0.5013 | 0.4971 | 0.7084 | 0.7348 | 0.7198 | 0.4314 | 0.0055 | 0.0886 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.4625 | 0.8975 | 0.8061 | 0.6363 | 0.9510 | 0.8995 | 0.8029 | 0.3095 | 0.7941 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4093 | 0.4557 | 0.4489 | 0.5702 | 0.6200 | 0.6199 | 0.6871 | 0.0373 | 0.4616 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.8036 | 0.4032 | 0.3818 | 0.8760 | 0.5936 | 0.5877 | 0.6057 | 0.0246 | 0.0189 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.2369 | 0.6114 | 0.5479 | 0.5931 | 0.8259 | 0.7795 | 0.8402 | 0.0092 | 0.4961 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2410 | 0.4210 | 0.3706 | 0.6092 | 0.7416 | 0.7170 | 0.4060 | 0.0021 | 0.0886 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.4827 | 0.5003 | 0.4952 | 0.7663 | 0.7571 | 0.7579 | 0.3505 | 0.0018 | 0.0635 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.4313 | 0.3800 | 0.4025 | 0.4348 | 0.3317 | 0.3989 | 0.3309 | 0.0333 | 0.0704 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3183 | 0.3862 | 0.4033 | 0.3389 | 0.3456 | 0.3586 | 0.5130 | 0.0495 | 0.1932 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5142 | 0.5925 | 0.6113 | 0.4647 | 0.4900 | 0.5539 | 0.4752 | 0.0451 | 0.1894 |
