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

This code has been edited extensively (you can see old code on deprecated branch).  Hopefully it is easier to use, but please bug me if you run into issues.

## Getting Started

1.  Please clone my git repo.  You will need to use my version of [caffe](https://github.com/LisaAnne/lisa-caffe-public/tree/bilinear), specifically the "bilinear" branch.
2.  Download data using the "download_data.sh" script.  This will also preprocess the CUB sentences.  All my ECCV 2016 models will be put in "gve_models"

## Building the models

All the models are generated using NetSpec.  Please build them by running "build_nets.sh".  "build_nets.sh" will also generate bash scripts you can use to train models.

## Training the models

If you would like to retrain my models, please use the following instructions.  Note that all my trained models are in "gve_models".  All the training scripts will be built using "build_nets.sh"

1.  First train the description model ("./train_description.sh").  The learned hidden units of the description model are used to build a representation for the 200 CUB classes.
2.  Run "make_class_embedding.sh" to build the class embeddings
3.  Train definition and explanation_label models ("./train_definition.sh", "./train_explanation_label.sh").
4.  Train sentence classification model ("./train_caption_classifer.sh").  This is needed for the reinforce loss.  I found that using an embedding and LSTM hidden dim of 1000 and dropout of 0.75 worked best.
5.  Train the explanation_dis and explanation models ("./train_explanation_dis.sh", "./train_explanation.sh").  These models are fine-tuned from the description and explanation_label model respecitvely.  The weighting between the relevance and discriminative loss can impact perforance substantially.  I found that loss weights of 80/20 on the relevance/discriminative losses worked best for the explanation-dis model and that loss weights of 110/20 on the relevance/discriminative losses worked best for the explanation model.  


## Evaluation models

Please use the bash scripts eval_*.sh to compute image relevance metrics.  To compute class relevance metrics, run "analyze_cider_scores.py".  This relies on precomputed CIDEr scores between each generated sentence and reference sentences from each class.  You can recompute these using "class_meteor_similarity_metric.py" but this will take > 10 hours.

Please note that I retrained the models since the initial arXiv version of the paper was released so the numbers are slightly different, though the main trends remain the same.
