# RAE-DiT IAD results (ab_noise_sigma005)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.05 | 1 | joint_best.pt | disc | 0.5670 | 0.5438 | 0.5461 | 0.7229 | 0.7068 | 0.7138 | 0.7387 | 0.0595 | 0.2685 |
| single | 0.05 | 1 | joint_best.pt | latent_cos | 0.3336 | 0.4168 | 0.3859 | 0.5876 | 0.6182 | 0.6027 | 0.2775 | 0.0128 | 0.0799 |
| single | 0.05 | 1 | joint_best.pt | pixel | 0.4755 | 0.6352 | 0.6486 | 0.6891 | 0.7901 | 0.7952 | 0.7431 | 0.1572 | 0.5523 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.05 | 1 | joint_best.pt | pixel | 0.3299 | 0.4045 | 0.4065 | 0.6859 | 0.7383 | 0.7356 | 0.3379 | 0.0135 | 0.0956 |
| carpet | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.0032 | 0.0694 | 0.0373 | 0.5546 | 0.5706 | 0.5607 | 0.0233 | 0.0081 | 0.0000 |
| carpet | single | 0.05 | 1 | joint_best.pt | disc | 0.7733 | 0.4952 | 0.5578 | 0.9330 | 0.8131 | 0.8508 | 0.7643 | 0.0396 | 0.2285 |
| grid | single | 0.05 | 1 | joint_best.pt | pixel | 0.7185 | 0.7803 | 0.8847 | 0.8668 | 0.9032 | 0.9517 | 0.8635 | 0.0861 | 0.7306 |
| grid | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.2047 | 0.3525 | 0.3074 | 0.5753 | 0.6649 | 0.6279 | 0.0184 | 0.0035 | 0.0000 |
| grid | single | 0.05 | 1 | joint_best.pt | disc | 0.5547 | 0.7118 | 0.6884 | 0.7551 | 0.8541 | 0.8503 | 0.6426 | 0.0099 | 0.1794 |
| hazelnut | single | 0.05 | 1 | joint_best.pt | pixel | 0.5564 | 0.9382 | 0.8986 | 0.7078 | 0.9659 | 0.9429 | 0.9601 | 0.5260 | 0.9148 |
| hazelnut | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.3721 | 0.4336 | 0.4239 | 0.5777 | 0.5822 | 0.5741 | 0.2074 | 0.0123 | 0.0092 |
| hazelnut | single | 0.05 | 1 | joint_best.pt | disc | 0.6800 | 0.6100 | 0.6093 | 0.8193 | 0.7289 | 0.7307 | 0.9061 | 0.1567 | 0.6435 |
| screw | single | 0.05 | 1 | joint_best.pt | pixel | 0.0439 | 0.5755 | 0.5515 | 0.5426 | 0.8336 | 0.8035 | 0.8656 | 0.0383 | 0.5726 |
| screw | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.9000 | 0.8649 | 0.8795 | 0.9627 | 0.9469 | 0.9518 | 0.7553 | 0.0046 | 0.2456 |
| screw | single | 0.05 | 1 | joint_best.pt | disc | 0.3789 | 0.5339 | 0.5112 | 0.6794 | 0.7723 | 0.7648 | 0.6451 | 0.0032 | 0.1413 |
| transistor | single | 0.05 | 1 | joint_best.pt | pixel | 0.7288 | 0.4775 | 0.5017 | 0.6423 | 0.5096 | 0.5421 | 0.6885 | 0.1221 | 0.4477 |
| transistor | single | 0.05 | 1 | joint_best.pt | latent_cos | 0.1879 | 0.3637 | 0.2812 | 0.2675 | 0.3265 | 0.2988 | 0.3829 | 0.0354 | 0.1449 |
| transistor | single | 0.05 | 1 | joint_best.pt | disc | 0.4479 | 0.3683 | 0.3638 | 0.4279 | 0.3656 | 0.3724 | 0.7354 | 0.0881 | 0.1499 |
