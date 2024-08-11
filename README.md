# FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process
## Introduction
The implementation of **FakingRecipe**, a creative process-aware model for detecting fake news short videos. It
captures the fake news preferences in material selection from sentimental and semantic aspects and considers the traits of material editing from spatial and temporal aspects.

[Preprint](https://www.arxiv.org/abs/2407.16670)
<!-- ## File Structure
```shell
.
├── README  # * Instruction to this repo
├── requirements  # * Requirements for Conda Environment
├── data  # * Place data split & preprocessed data
├── models  # * Codes for FakingRecipe Model
├── utils  # * Codes for Training and Inference
├── main  # * Codes for Training and Inference
└── run  # * Codes for Training and Inference
    
``` -->

## Dataset
We conduct experiments on two datasets: FakeSV and FakeTT. 
### FakeSV
FakeSV is the largest publicly available Chinese dataset for fake news detection on short video platforms, featuring samples from
Douyin and Kuaishou, two popular Chinese short video platforms. Each sample in FakeSV contains the video itself, its title, comments, metadata, and publisher profiles. For the details, please refer to [this repo](https://github.com/ICTMCG/FakeSV).
### FakeTT
FakeTT is our newly constructed English dataset for a comprehensive evaluation in English-speaking contexts. 
- **Collection**: 
We utilized the well-known fact-checking website Snopes as our primary source for identifying potential fake news events in multiple domains. We filtered reports published between January 2018 and January 2024, using the keywords “video” and “TikTok” to retrieve video-form fake news instances on TikTok. We extracted descriptions of 365
verified fake news events from these Snopes reports to use as search queries on TikTok. 
- **Annotation**:
We manually annotated each collected video to assess its veracity. Each video underwent rigorous scrutiny by at least two independent annotators and was classified as “fake”, “real”, or “uncertain”. The annotation process yielded 1,336 fake news videos and 867 real news videos. After further filtering to include only videos shorter than three minutes, we formed the FakeTT dataset. FakeTT encompasses 286 news events, comprising 1,172 fake and 819 real news videos. 
- **Data Format**:
  ```
    {
        "video_id":"7299305894641208607",
        "description":"putin and Kim Jung are either both socislly ackward or hoth have trust issues! #funnypolitics #trump2024 #politicalhumor ",
        "annotation":"fake",
        "user_certify":0,  # 1 if the account is verified else 0
        "user_description":"Your business, relationship and life coach!",
        "publish_time":1699502099000,
        "event":"trust issues Kim Putin"
    }
  ```
- **Data Acquisition**
If you would like to access the FakeTT dataset, please fill out this [Application Form](https://forms.office.com/Pages/ResponsePage.aspx?id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAO__R5hy59UMEEyNENDVTlYMzZSRjlQQkIzRFg3TEpIMy4u). The download link will be sent to you once the form is accepted.

## Data Preprocess
- To extract OCR, we use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
- To seg video clips, we use [TransNetv2](https://github.com/soCzech/TransNetV2).
- To facilitate reproduction, we provide preprocessed features, which you can download from [this link]() and place the '/fea' directory under FakingRecipe (at the same level as main.py). Additionally, we offer [checkpoints]() for two datasets, which you can similarly place the '/provided_ckp' directory under FakingRecipe.

## Quick Start
You can utilize FakeRecipe to infer the authenticity of the samples from the test set by following code:
 ```
 # Infer the examples from FakeSV
  python main.py  --dataset fakesv  --mode inference_test --inference_ckp ./provided_ckp/FakingRecipe_fakesv

  # Infer the examples from FakeTT
  python main.py  --dataset fakett  --mode inference_test --inference_ckp ./provided_ckp/FakingRecipe_fakett
  ```


## Citation
If you find our dataset and code are helpful, please cite the following ACM MM 2024 paper:
 ```
@inproceedings{fakingrecipe,
title={FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process},
author={Bu, Yuyan and Sheng, Qiang and Cao, Juan and Qi, Peng and Wang, Danding and Li, Jintao},
booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
year={2024},
doi={10.1145/3664647.3680663},
publisher = {Association for Computing Machinery},
}
  ```