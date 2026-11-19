# RAE-DiT IAD results (J5_latent_mix)

Aggregate = mean over categories (ckpt = best / last). Image AUROC/AP reported
for all three aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | best | disc | 0.7470 | 0.9235 | 0.9204 | 0.9007 | 0.9697 | 0.9688 | 0.9359 | 0.5457 | 0.8336 |
| single | 1.0 | 1 | best | latent_cos | 0.9613 | 0.9735 | 0.9746 | 0.9816 | 0.9871 | 0.9885 | 0.9389 | 0.3525 | 0.8507 |
| single | 1.0 | 1 | best | pixel | 0.5710 | 0.6893 | 0.6679 | 0.7875 | 0.8443 | 0.8398 | 0.6942 | 0.1600 | 0.4553 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bottle | single | 1.0 | 1 | bottle_best.pt | pixel | 0.8381 | 0.9190 | 0.9413 | 0.9454 | 0.9758 | 0.9837 | 0.8471 | 0.4735 | 0.6515 |
| bottle | single | 1.0 | 1 | bottle_best.pt | latent_cos | 1.0000 | 0.9992 | 1.0000 | 1.0000 | 0.9998 | 1.0000 | 0.9502 | 0.5908 | 0.8456 |
| bottle | single | 1.0 | 1 | bottle_best.pt | disc | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9880 | 0.8294 | 0.9500 |
| cable | single | 1.0 | 1 | cable_best.pt | pixel | 0.5611 | 0.4906 | 0.5270 | 0.7318 | 0.6339 | 0.6560 | 0.5155 | 0.0354 | 0.2953 |
| cable | single | 1.0 | 1 | cable_best.pt | latent_cos | 0.8891 | 0.9183 | 0.9228 | 0.9425 | 0.9538 | 0.9590 | 0.9036 | 0.2900 | 0.7456 |
| cable | single | 1.0 | 1 | cable_best.pt | disc | 0.9202 | 0.9307 | 0.9286 | 0.9553 | 0.9645 | 0.9640 | 0.9353 | 0.5478 | 0.7681 |
| capsule | single | 1.0 | 1 | capsule_best.pt | pixel | 0.7188 | 0.7327 | 0.7252 | 0.9193 | 0.9255 | 0.9221 | 0.3143 | 0.0302 | 0.0480 |
| capsule | single | 1.0 | 1 | capsule_best.pt | latent_cos | 0.9446 | 0.9501 | 0.9442 | 0.9877 | 0.9890 | 0.9876 | 0.9722 | 0.2999 | 0.8399 |
| capsule | single | 1.0 | 1 | capsule_best.pt | disc | 0.8440 | 0.8404 | 0.8428 | 0.9628 | 0.9630 | 0.9632 | 0.9601 | 0.3887 | 0.8137 |
| carpet | single | 1.0 | 1 | carpet_best.pt | pixel | 0.6810 | 0.6685 | 0.7368 | 0.8891 | 0.8691 | 0.9046 | 0.6702 | 0.1045 | 0.3141 |
| carpet | single | 1.0 | 1 | carpet_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9791 | 0.3744 | 0.9103 |
| carpet | single | 1.0 | 1 | carpet_best.pt | disc | 0.2949 | 0.9912 | 0.9876 | 0.7480 | 0.9972 | 0.9961 | 0.9886 | 0.7347 | 0.9539 |
| grid | single | 1.0 | 1 | grid_best.pt | pixel | 0.4921 | 0.6500 | 0.5689 | 0.7221 | 0.8561 | 0.8019 | 0.7546 | 0.1157 | 0.4611 |
| grid | single | 1.0 | 1 | grid_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9862 | 0.2898 | 0.9481 |
| grid | single | 1.0 | 1 | grid_best.pt | disc | 0.2556 | 0.9457 | 0.9315 | 0.6945 | 0.9820 | 0.9769 | 0.9804 | 0.2164 | 0.9154 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | pixel | 0.5189 | 0.9621 | 0.9154 | 0.6937 | 0.9808 | 0.9559 | 0.8960 | 0.5201 | 0.8664 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | latent_cos | 0.9907 | 0.9786 | 0.9804 | 0.9943 | 0.9848 | 0.9865 | 0.9810 | 0.4337 | 0.9230 |
| hazelnut | single | 1.0 | 1 | hazelnut_best.pt | disc | 0.7818 | 0.9014 | 0.8964 | 0.8862 | 0.9514 | 0.9496 | 0.9811 | 0.7646 | 0.8829 |
| leather | single | 1.0 | 1 | leather_best.pt | pixel | 0.4810 | 0.8577 | 0.7588 | 0.7991 | 0.9510 | 0.9145 | 0.7664 | 0.1317 | 0.5861 |
| leather | single | 1.0 | 1 | leather_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9268 | 0.2047 | 0.8495 |
| leather | single | 1.0 | 1 | leather_best.pt | disc | 0.6437 | 0.9212 | 0.9222 | 0.8772 | 0.9744 | 0.9749 | 0.9259 | 0.3439 | 0.7720 |
| metal_nut | single | 1.0 | 1 | metal_nut_best.pt | pixel | 0.4565 | 0.6002 | 0.5582 | 0.7602 | 0.8741 | 0.8484 | 0.6030 | 0.1811 | 0.4125 |
| metal_nut | single | 1.0 | 1 | metal_nut_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8625 | 0.3748 | 0.8517 |
| metal_nut | single | 1.0 | 1 | metal_nut_best.pt | disc | 0.9731 | 1.0000 | 1.0000 | 0.9943 | 1.0000 | 1.0000 | 0.9001 | 0.6311 | 0.9089 |
| pill | single | 1.0 | 1 | pill_best.pt | pixel | 0.5881 | 0.5966 | 0.5693 | 0.8764 | 0.8883 | 0.8722 | 0.7761 | 0.0815 | 0.5546 |
| pill | single | 1.0 | 1 | pill_best.pt | latent_cos | 0.9618 | 0.9523 | 0.9615 | 0.9930 | 0.9897 | 0.9919 | 0.9609 | 0.4033 | 0.9160 |
| pill | single | 1.0 | 1 | pill_best.pt | disc | 0.8227 | 0.9070 | 0.9026 | 0.9653 | 0.9814 | 0.9809 | 0.9000 | 0.5704 | 0.8275 |
| screw | single | 1.0 | 1 | screw_best.pt | pixel | 0.2453 | 0.6376 | 0.5841 | 0.5964 | 0.8459 | 0.8104 | 0.8654 | 0.0218 | 0.5583 |
| screw | single | 1.0 | 1 | screw_best.pt | latent_cos | 0.7467 | 0.9108 | 0.9037 | 0.8963 | 0.9700 | 0.9674 | 0.9809 | 0.2724 | 0.8985 |
| screw | single | 1.0 | 1 | screw_best.pt | disc | 0.8221 | 0.8024 | 0.8125 | 0.9228 | 0.9269 | 0.9301 | 0.8844 | 0.1648 | 0.6792 |
| tile | single | 1.0 | 1 | tile_best.pt | pixel | 0.5808 | 0.7006 | 0.6811 | 0.8106 | 0.8634 | 0.8619 | 0.6615 | 0.2190 | 0.3733 |
| tile | single | 1.0 | 1 | tile_best.pt | latent_cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8665 | 0.3721 | 0.7181 |
| tile | single | 1.0 | 1 | tile_best.pt | disc | 0.9737 | 1.0000 | 1.0000 | 0.9914 | 1.0000 | 1.0000 | 0.9833 | 0.8892 | 0.9138 |
| toothbrush | single | 1.0 | 1 | toothbrush_best.pt | pixel | 0.5694 | 0.5472 | 0.6139 | 0.8179 | 0.7220 | 0.7848 | 0.8763 | 0.0898 | 0.6504 |
| toothbrush | single | 1.0 | 1 | toothbrush_best.pt | latent_cos | 0.9444 | 0.9444 | 0.9389 | 0.9748 | 0.9762 | 0.9733 | 0.9745 | 0.2390 | 0.8669 |
| toothbrush | single | 1.0 | 1 | toothbrush_best.pt | disc | 0.9194 | 0.9778 | 0.9722 | 0.9746 | 0.9918 | 0.9898 | 0.9870 | 0.5540 | 0.8268 |
| transistor | single | 1.0 | 1 | transistor_best.pt | pixel | 0.4683 | 0.4442 | 0.4617 | 0.4663 | 0.4181 | 0.4795 | 0.3790 | 0.0420 | 0.1508 |
| transistor | single | 1.0 | 1 | transistor_best.pt | latent_cos | 0.9504 | 0.9537 | 0.9738 | 0.9386 | 0.9448 | 0.9644 | 0.8905 | 0.3454 | 0.7186 |
| transistor | single | 1.0 | 1 | transistor_best.pt | disc | 0.8887 | 0.8975 | 0.8992 | 0.8561 | 0.8953 | 0.8993 | 0.7503 | 0.3690 | 0.6777 |
| wood | single | 1.0 | 1 | wood_best.pt | pixel | 0.6202 | 0.8675 | 0.8333 | 0.8580 | 0.9579 | 0.9468 | 0.6649 | 0.2275 | 0.3588 |
| wood | single | 1.0 | 1 | wood_best.pt | latent_cos | 0.9947 | 0.9956 | 0.9947 | 0.9984 | 0.9986 | 0.9983 | 0.8917 | 0.4915 | 0.8652 |
| wood | single | 1.0 | 1 | wood_best.pt | disc | 0.3886 | 0.7807 | 0.7491 | 0.7715 | 0.9296 | 0.9172 | 0.9242 | 0.6060 | 0.7321 |
| zipper | single | 1.0 | 1 | zipper_best.pt | pixel | 0.7453 | 0.6641 | 0.5441 | 0.9256 | 0.9016 | 0.8538 | 0.8229 | 0.1254 | 0.5482 |
| zipper | single | 1.0 | 1 | zipper_best.pt | latent_cos | 0.9966 | 0.9987 | 0.9987 | 0.9991 | 0.9996 | 0.9996 | 0.9571 | 0.3063 | 0.8638 |
| zipper | single | 1.0 | 1 | zipper_best.pt | disc | 0.6762 | 0.9559 | 0.9606 | 0.9112 | 0.9887 | 0.9897 | 0.9499 | 0.5750 | 0.8820 |
