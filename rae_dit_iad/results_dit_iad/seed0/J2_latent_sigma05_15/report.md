# RAE-DiT IAD results (J2_latent_sigma05_15)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 1.0 | 1 | joint_best.pt | disc | 0.8824 | 0.8144 | 0.8347 | 0.9119 | 0.8702 | 0.8816 | 0.8123 | 0.1774 | 0.7239 |
| single | 1.0 | 1 | joint_best.pt | latent_cos | 0.8693 | 0.8145 | 0.8498 | 0.9116 | 0.8689 | 0.8876 | 0.9262 | 0.2088 | 0.6539 |
| single | 1.0 | 1 | joint_best.pt | pixel | 0.4125 | 0.5683 | 0.5431 | 0.6373 | 0.7309 | 0.7256 | 0.5704 | 0.0741 | 0.3327 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 1.0 | 1 | joint_best.pt | pixel | 0.5068 | 0.5088 | 0.5562 | 0.8200 | 0.8052 | 0.8382 | 0.3394 | 0.0113 | 0.0668 |
| carpet | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.9679 | 0.9559 | 0.9699 | 0.9908 | 0.9865 | 0.9907 | 0.8972 | 0.0923 | 0.5505 |
| carpet | single | 1.0 | 1 | joint_best.pt | disc | 0.9952 | 0.7713 | 0.7994 | 0.9985 | 0.8821 | 0.9038 | 0.7099 | 0.0762 | 0.6516 |
| grid | single | 1.0 | 1 | joint_best.pt | pixel | 0.3977 | 0.4302 | 0.3935 | 0.6897 | 0.7295 | 0.6881 | 0.5284 | 0.0073 | 0.2146 |
| grid | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.9933 | 0.9841 | 0.9908 | 0.9976 | 0.9943 | 0.9968 | 0.9425 | 0.1253 | 0.7583 |
| grid | single | 1.0 | 1 | joint_best.pt | disc | 0.9900 | 0.9591 | 0.9733 | 0.9965 | 0.9855 | 0.9904 | 0.9139 | 0.1849 | 0.8427 |
| hazelnut | single | 1.0 | 1 | joint_best.pt | pixel | 0.4832 | 0.8904 | 0.7939 | 0.6469 | 0.9475 | 0.8948 | 0.8072 | 0.3084 | 0.7866 |
| hazelnut | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.9279 | 0.8575 | 0.8868 | 0.9627 | 0.9139 | 0.9329 | 0.9783 | 0.4564 | 0.8477 |
| hazelnut | single | 1.0 | 1 | joint_best.pt | disc | 0.9889 | 0.9632 | 0.9818 | 0.9950 | 0.9816 | 0.9908 | 0.9440 | 0.3618 | 0.9196 |
| screw | single | 1.0 | 1 | joint_best.pt | pixel | 0.2503 | 0.6257 | 0.5686 | 0.5977 | 0.8323 | 0.7981 | 0.8491 | 0.0100 | 0.5191 |
| screw | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.6412 | 0.5702 | 0.6557 | 0.8286 | 0.8059 | 0.8425 | 0.9439 | 0.0833 | 0.5270 |
| screw | single | 1.0 | 1 | joint_best.pt | disc | 0.7243 | 0.7391 | 0.7430 | 0.8806 | 0.9011 | 0.9002 | 0.8874 | 0.1436 | 0.6914 |
| transistor | single | 1.0 | 1 | joint_best.pt | pixel | 0.4246 | 0.3862 | 0.4033 | 0.4323 | 0.3400 | 0.4088 | 0.3282 | 0.0335 | 0.0767 |
| transistor | single | 1.0 | 1 | joint_best.pt | latent_cos | 0.8163 | 0.7050 | 0.7458 | 0.7783 | 0.6436 | 0.6750 | 0.8688 | 0.2866 | 0.5863 |
| transistor | single | 1.0 | 1 | joint_best.pt | disc | 0.7133 | 0.6392 | 0.6763 | 0.6888 | 0.6007 | 0.6230 | 0.6064 | 0.1207 | 0.5140 |
