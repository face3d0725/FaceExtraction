# FaceExtraction

Code and dataset for 

FaceOcc: A Diverse, High-quality Face Occlusion Dataset for Human Face Extraction

# Requirements
PyTorch > 1.6.0

[Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)

PIL

cv2

numpy 

# How to use 
1. Download CelebAMask-HQ dataset, detect the facial landmarks using [3DDFAv2](https://github.com/cleardusk/3DDFA_V2)
2. Specify the directories in `face_align/process_CelebAMaskHQ.py`
3. Run `face_align/process_CelebAMaskHQ.py` to generate&align CelebAMask-HQ images and masks
4.Download FaceOcc and put it under Dataset directory
5.Run train.py
