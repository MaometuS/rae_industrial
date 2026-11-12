# RAE-DiT IAD results (K_latent_simplenet_adaptor)

Aggregate = mean over categories. Image AUROC/AP reported for all three
aggregations (mean / max / top-1% pixel scores).

## Aggregate

| mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 0.1 | 1 | joint_best.pt | disc | 0.5753 | 0.4859 | 0.4756 | 0.7368 | 0.6626 | 0.6636 | 0.4830 | 0.0178 | 0.0663 |
| single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5172 | 0.4501 | 0.4664 | 0.6791 | 0.6376 | 0.6490 | 0.6475 | 0.0393 | 0.3652 |
| single | 0.1 | 1 | joint_best.pt | pixel | 0.4054 | 0.5709 | 0.5429 | 0.6336 | 0.7296 | 0.7219 | 0.5673 | 0.0759 | 0.3234 |

## Per-category

| cat | mode | t | reps | ckpt | score | img_auroc_mean | img_auroc_max | img_auroc_top1 | img_ap_mean | img_ap_max | img_ap_top1 | px_auroc | px_ap | aupro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| carpet | single | 0.1 | 1 | joint_best.pt | pixel | 0.5016 | 0.5237 | 0.5522 | 0.8156 | 0.8100 | 0.8353 | 0.3452 | 0.0115 | 0.0684 |
| carpet | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.5490 | 0.4807 | 0.4141 | 0.8221 | 0.7782 | 0.7290 | 0.8624 | 0.0800 | 0.5732 |
| carpet | single | 0.1 | 1 | joint_best.pt | disc | 0.5558 | 0.4450 | 0.4623 | 0.8256 | 0.7360 | 0.7550 | 0.6922 | 0.0235 | 0.0956 |
| grid | single | 0.1 | 1 | joint_best.pt | pixel | 0.4010 | 0.4394 | 0.4069 | 0.6911 | 0.7234 | 0.6937 | 0.5244 | 0.0072 | 0.2040 |
| grid | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.8505 | 0.3776 | 0.5355 | 0.9271 | 0.6581 | 0.7617 | 0.8799 | 0.0390 | 0.6679 |
| grid | single | 0.1 | 1 | joint_best.pt | disc | 0.5263 | 0.5372 | 0.5221 | 0.7460 | 0.7528 | 0.7342 | 0.4320 | 0.0055 | 0.0842 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | pixel | 0.4621 | 0.8982 | 0.8068 | 0.6359 | 0.9523 | 0.9020 | 0.7983 | 0.3185 | 0.7874 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3961 | 0.4400 | 0.4450 | 0.5625 | 0.6003 | 0.6141 | 0.6081 | 0.0268 | 0.3113 |
| hazelnut | single | 0.1 | 1 | joint_best.pt | disc | 0.7718 | 0.4207 | 0.3643 | 0.8501 | 0.6086 | 0.5856 | 0.6333 | 0.0269 | 0.0352 |
| screw | single | 0.1 | 1 | joint_best.pt | pixel | 0.2318 | 0.6153 | 0.5458 | 0.5915 | 0.8310 | 0.7798 | 0.8366 | 0.0090 | 0.4883 |
| screw | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.4505 | 0.5569 | 0.5384 | 0.7477 | 0.8036 | 0.7980 | 0.3779 | 0.0018 | 0.0782 |
| screw | single | 0.1 | 1 | joint_best.pt | disc | 0.4928 | 0.4960 | 0.5003 | 0.7751 | 0.7719 | 0.7699 | 0.3427 | 0.0017 | 0.0537 |
| transistor | single | 0.1 | 1 | joint_best.pt | pixel | 0.4304 | 0.3779 | 0.4029 | 0.4339 | 0.3316 | 0.3986 | 0.3318 | 0.0333 | 0.0690 |
| transistor | single | 0.1 | 1 | joint_best.pt | latent_cos | 0.3400 | 0.3954 | 0.3992 | 0.3359 | 0.3479 | 0.3423 | 0.5094 | 0.0492 | 0.1956 |
| transistor | single | 0.1 | 1 | joint_best.pt | disc | 0.5296 | 0.5304 | 0.5288 | 0.4872 | 0.4438 | 0.4736 | 0.3147 | 0.0316 | 0.0628 |
