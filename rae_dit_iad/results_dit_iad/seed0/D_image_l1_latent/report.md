# RAE-DiT IAD results (D_image_l1_latent)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5115 | 0.5017 | 0.5206 | 0.6999 | 0.7043 | 0.7183 | 0.7131 | 0.0645 | 0.2362 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4245 | 0.5048 | 0.4653 | 0.6317 | 0.6994 | 0.6646 | 0.7142 | 0.0305 | 0.4087 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.5595 | 0.6426 | 0.6590 | 0.7478 | 0.7801 | 0.7927 | 0.7359 | 0.1470 | 0.5178 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5791 | 0.5795 | 0.6003 | 0.8216 | 0.8134 | 0.8142 | 0.3841 | 0.0155 | 0.1115 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2468 | 0.4378 | 0.2869 | 0.6226 | 0.7609 | 0.6467 | 0.8046 | 0.0408 | 0.5058 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.6746 | 0.4675 | 0.4807 | 0.9019 | 0.7920 | 0.8059 | 0.7425 | 0.0346 | 0.1835 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.7185 | 0.7068 | 0.8179 | 0.8885 | 0.8800 | 0.9248 | 0.8444 | 0.0573 | 0.6581 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4528 | 0.4060 | 0.3584 | 0.7296 | 0.7177 | 0.6431 | 0.6220 | 0.0100 | 0.2972 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.3926 | 0.6383 | 0.6625 | 0.7225 | 0.8274 | 0.8506 | 0.5276 | 0.0066 | 0.0308 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.7368 | 0.9246 | 0.8829 | 0.8409 | 0.9611 | 0.9372 | 0.9686 | 0.5422 | 0.9070 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4986 | 0.5043 | 0.5804 | 0.6330 | 0.6547 | 0.7026 | 0.7580 | 0.0417 | 0.4191 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7107 | 0.5246 | 0.5671 | 0.8311 | 0.6965 | 0.7224 | 0.9011 | 0.1727 | 0.6027 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.0978 | 0.5222 | 0.5298 | 0.5583 | 0.7943 | 0.7915 | 0.8468 | 0.0230 | 0.5208 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.6559 | 0.5110 | 0.5591 | 0.8287 | 0.7492 | 0.7903 | 0.8970 | 0.0125 | 0.6564 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.3571 | 0.4819 | 0.4821 | 0.6589 | 0.7708 | 0.7812 | 0.6617 | 0.0034 | 0.1818 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.6654 | 0.4800 | 0.4642 | 0.6298 | 0.4516 | 0.4957 | 0.6353 | 0.0968 | 0.3914 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.2683 | 0.6650 | 0.5417 | 0.3445 | 0.6146 | 0.5405 | 0.4895 | 0.0474 | 0.1648 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.4225 | 0.3962 | 0.4104 | 0.3851 | 0.4350 | 0.4313 | 0.7328 | 0.1051 | 0.1824 |
