# Latent DRAEM pilot

Mean over categories. `disc` = learned discriminator map, `resid` = cos(z, R(z)), `knn` = frozen-token kNN baseline (no training).

## DINOv2-B

|                          |   img_auroc |   img_auroc_top1 |   px_auroc |   px_ap |   aupro |
|:-------------------------|------------:|-----------------:|-----------:|--------:|--------:|
| ('-', 'knn')             |       0.907 |            0.889 |      0.901 |   0.233 |   0.755 |
| ('latent_self', 'disc')  |       0.677 |            0.679 |      0.8   |   0.009 |   0.412 |
| ('latent_self', 'resid') |       0.375 |            0.373 |      0.821 |   0.008 |   0.484 |
| ('mix', 'disc')          |       0.706 |            0.688 |      0.8   |   0.153 |   0.6   |
| ('mix', 'resid')         |       0.775 |            0.762 |      0.917 |   0.098 |   0.683 |
| ('pixel', 'disc')        |       0.736 |            0.722 |      0.97  |   0.284 |   0.854 |
| ('pixel', 'resid')       |       0.771 |            0.764 |      0.92  |   0.179 |   0.668 |

## img_auroc per category (disc, and knn baseline)

| category   |   ('DINOv2-B', '-') |   ('DINOv2-B', 'latent_self') |   ('DINOv2-B', 'mix') |   ('DINOv2-B', 'pixel') |
|:-----------|--------------------:|------------------------------:|----------------------:|------------------------:|
| screw      |               0.907 |                         0.677 |                 0.706 |                   0.736 |

## px_auroc per category (disc, and knn baseline)

| category   |   ('DINOv2-B', '-') |   ('DINOv2-B', 'latent_self') |   ('DINOv2-B', 'mix') |   ('DINOv2-B', 'pixel') |
|:-----------|--------------------:|------------------------------:|----------------------:|------------------------:|
| screw      |               0.901 |                           0.8 |                   0.8 |                    0.97 |

## aupro per category (disc, and knn baseline)

| category   |   ('DINOv2-B', '-') |   ('DINOv2-B', 'latent_self') |   ('DINOv2-B', 'mix') |   ('DINOv2-B', 'pixel') |
|:-----------|--------------------:|------------------------------:|----------------------:|------------------------:|
| screw      |               0.755 |                         0.412 |                   0.6 |                   0.854 |
