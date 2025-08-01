# SinSEMI
This is the private code base of SinSEM, the paper is currently reviewed and the code is verified by AAAI.
## Requirements
```bash
python -m pip install -r requirements.txt
```
This code was built and tested with python 3.8 and torch 2.1.0
## SinSEMI Training
The training images are provided in ./data/training_data/.
```
python main.py --mode train \
                --dataset_folder ./data/training_data/<Line_Pair> \ 
                --image_name <line_pair_bridge_defect.png> \
                --results_folder ./results/ 
```

## SinSEMI Sampling
Generate samples using a trained SinSEMI model:
```
python main.py --mode sample \ 
                --dataset_folder ./data/training_data/<Line_Pair> \
                --image_name <line_pair_bridge_defect.png> \
                --results_folder ./results/ \ 
                --load_milestone 12
```
If use LPIPS energy guidance, add `--lpips_guidance`
## SIFID & LPIPS Evaluation
The SIFID and LPIPS are calculated using [lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch). Install required packages before running:
```
# calculate sifid score
python sifid_lpips/sifid_score.py --path2real <folder_path_to_real> --path2fake <folder_path_to_fake>
# calculate lpips score
python sifid_lpips/lpips.py --path2real <folder_path_to_real> --path2fake <folder_path_to_fake>
```
## Segmentation Evaluation 
The segmentaion code is provided as followed.
```
python segmentation/segmentation_evaluation.py --path2real-defect <file_path_to_real_with_defect> --path2real-nodefect <file_path_to_real_without_defect> --path2fake <folder_path_to_fake>
```
All the segmentation pretrained models and testing outputs are provided in [segmentation_results_link](https://drive.google.com/file/d/1zJUb9eco9ul2eduSmYJ0JpWvlkLx39DW/view?usp=sharing).
## Training data and Evaluation Data
The training images are provided in ./data/training_data/.

All the evaluation images in the paper are provided in ./data/evaluation_data as .zip files. 

All images are provide in PNG format.
## Pretrained Models
All pretrained models are in [pretrain_models_zip_link](https://drive.google.com/file/d/1sltmYL0K2NRw2-CXjw52gFDXPfQCNC3L/view?usp=sharing).
## Sourse
The SinSEMI code was adapted from the following [SinDDM](https://github.com/fallenshock/SinDDM)
