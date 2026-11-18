# RAE-DiT IAD results (J_latent_simplenet_nodecoder)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | best | disc | 0.6657 | 0.7599 | 0.7553 | 0.7740 | 0.8279 | 0.8258 | 0.7831 | 0.1538 | 0.4575 |
| single | 0.1 | 1 | best | latent_cos | 0.8798 | 0.8503 | 0.8688 | 0.9260 | 0.9065 | 0.9242 | 0.8758 | 0.2500 | 0.5383 |
| single | 0.1 | 1 | best | pixel | 0.4187 | 0.5683 | 0.5460 | 0.6389 | 0.7308 | 0.7266 | 0.5702 | 0.0744 | 0.3318 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | carpet_best.pt | pixel | 0.5084 | 0.5116 | 0.5574 | 0.8216 | 0.8070 | 0.8387 | 0.3410 | 0.0114 | 0.0686 |
| carpet | single | 0.1 | 1 | carpet_best.pt | latent_cos | 0.9418 | 0.9330 | 0.9386 | 0.9834 | 0.9796 | 0.9822 | 0.8759 | 0.0976 | 0.5092 |
| carpet | single | 0.1 | 1 | carpet_best.pt | disc | 0.9109 | 0.9526 | 0.9482 | 0.9710 | 0.9873 | 0.9861 | 0.7576 | 0.0538 | 0.5022 |
| grid | single | 0.1 | 1 | grid_best.pt | pixel | 0.3960 | 0.4277 | 0.3926 | 0.6880 | 0.7258 | 0.6890 | 0.5265 | 0.0073 | 0.2116 |
| grid | single | 0.1 | 1 | grid_best.pt | latent_cos | 0.9616 | 0.9273 | 0.9457 | 0.9857 | 0.9763 | 0.9826 | 0.8899 | 0.0988 | 0.6803 |
| grid | single | 0.1 | 1 | grid_best.pt | disc | 0.5564 | 0.8931 | 0.8855 | 0.7516 | 0.9615 | 0.9584 | 0.8073 | 0.1078 | 0.5930 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | pixel | 0.4793 | 0.8914 | 0.7964 | 0.6449 | 0.9485 | 0.8965 | 0.8084 | 0.3096 | 0.7850 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | latent_cos | 0.9175 | 0.8754 | 0.8946 | 0.9573 | 0.9377 | 0.9501 | 0.9504 | 0.5623 | 0.6964 |
| hazelnut | single | 0.1 | 1 | hazelnut_best.pt | disc | 0.7521 | 0.7868 | 0.7746 | 0.8574 | 0.8836 | 0.8727 | 0.9561 | 0.4170 | 0.7207 |
| screw | single | 0.1 | 1 | screw_best.pt | pixel | 0.2880 | 0.6385 | 0.5909 | 0.6122 | 0.8422 | 0.8130 | 0.8500 | 0.0106 | 0.5221 |
| screw | single | 0.1 | 1 | screw_best.pt | latent_cos | 0.7543 | 0.6878 | 0.7102 | 0.8933 | 0.8497 | 0.8699 | 0.7630 | 0.0152 | 0.3423 |
| screw | single | 0.1 | 1 | screw_best.pt | disc | 0.5643 | 0.5427 | 0.5474 | 0.7665 | 0.7614 | 0.7650 | 0.6427 | 0.0033 | 0.1651 |
| transistor | single | 0.1 | 1 | transistor_best.pt | pixel | 0.4217 | 0.3721 | 0.3925 | 0.4276 | 0.3303 | 0.3958 | 0.3249 | 0.0330 | 0.0717 |
| transistor | single | 0.1 | 1 | transistor_best.pt | latent_cos | 0.8237 | 0.8279 | 0.8550 | 0.8103 | 0.7891 | 0.8362 | 0.9000 | 0.4763 | 0.4630 |
| transistor | single | 0.1 | 1 | transistor_best.pt | disc | 0.5450 | 0.6242 | 0.6208 | 0.5234 | 0.5459 | 0.5471 | 0.7520 | 0.1869 | 0.3064 |
