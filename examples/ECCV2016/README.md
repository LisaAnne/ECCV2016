# Generationg Visual Explanations 

This repository contains code for the following paper:

Hendricks, L.A., Akata, Z., Rohrbach, M., Donahue, J., Schiele, B. and Darrell, T., 2016. Generating Visual Explanations. ECCV 2016.

```
@article{hendricks2016generating,
  title={Generating Visual Explanations},
  author={Hendricks, Lisa Anne and Akata, Zeynep and Rohrbach, Marcus and Donahue, Jeff and Schiele, Bernt and Darrell, Trevor},
  journal={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2016}
}
```

## Getting Started

1.  Please clone my git repo.  My code for generating visual explanations is in /examples/ECCV2016.  You will need to use my version of caffe.
2.  Download data using the "download_data.sh" script.  This will also preprocess the CUB sentences.  All my ECCV 2016 models will be put in "gve_models"

## Building the models

All the models are generated using NetSpec.  Please build them by running "build_nets.sh".  "build_nets.sh" will also generate bash scripts you can use to train models.

## Evaluation models

Please use the bash scripts eval_*.sh to evaluate all the models.  Please note that I retrained the models since the initial arXiv version of the paper was released so the numbers are slightly different, though the main trends remain the same.
