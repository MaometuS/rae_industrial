# RAE-DiT IAD results (F_image_l1_adaptor)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5815 | 0.4837 | 0.4605 | 0.7334 | 0.6539 | 0.6446 | 0.4983 | 0.0162 | 0.0622 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4472 | 0.4691 | 0.4566 | 0.6402 | 0.6409 | 0.6325 | 0.3513 | 0.0142 | 0.0778 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4241 | 0.5590 | 0.5409 | 0.6441 | 0.7196 | 0.7178 | 0.6082 | 0.0722 | 0.3575 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5465 | 0.5433 | 0.5791 | 0.8408 | 0.8225 | 0.8529 | 0.3743 | 0.0124 | 0.0811 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3038 | 0.5859 | 0.5201 | 0.7093 | 0.8173 | 0.8041 | 0.5030 | 0.0149 | 0.0910 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.5987 | 0.4061 | 0.3624 | 0.8492 | 0.7151 | 0.7080 | 0.6780 | 0.0223 | 0.0662 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.3709 | 0.4135 | 0.3826 | 0.6825 | 0.7132 | 0.6875 | 0.5556 | 0.0078 | 0.2145 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.7109 | 0.4269 | 0.4670 | 0.8634 | 0.6960 | 0.6997 | 0.6367 | 0.0105 | 0.2456 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.4929 | 0.5305 | 0.5004 | 0.7259 | 0.7337 | 0.6962 | 0.4884 | 0.0062 | 0.0894 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.5086 | 0.8700 | 0.7914 | 0.6706 | 0.9253 | 0.8757 | 0.8985 | 0.2938 | 0.8499 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4511 | 0.4696 | 0.4411 | 0.5911 | 0.5982 | 0.5829 | 0.1412 | 0.0117 | 0.0001 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7071 | 0.3886 | 0.3493 | 0.8103 | 0.5736 | 0.5550 | 0.4425 | 0.0173 | 0.0008 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.2777 | 0.5841 | 0.5546 | 0.6079 | 0.8042 | 0.7845 | 0.8586 | 0.0115 | 0.5464 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4341 | 0.4841 | 0.4591 | 0.7150 | 0.7613 | 0.7372 | 0.1485 | 0.0014 | 0.0002 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.5636 | 0.5983 | 0.5954 | 0.7955 | 0.8256 | 0.8277 | 0.5388 | 0.0024 | 0.0922 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.4167 | 0.3837 | 0.3967 | 0.4185 | 0.3328 | 0.3885 | 0.3541 | 0.0355 | 0.0956 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3362 | 0.3787 | 0.3958 | 0.3220 | 0.3319 | 0.3387 | 0.3272 | 0.0325 | 0.0524 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5450 | 0.4950 | 0.4950 | 0.4860 | 0.4216 | 0.4362 | 0.3441 | 0.0330 | 0.0626 |
