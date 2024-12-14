

## Environment

- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- timm
- transformers
- h5py

## Directory Setup

<!---------------------------------------------------------------------------------------------------------------->
The structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataset/`
    - `bertvocab/`
        - `v2` : bert tokernizer
    - `EndoVis-18-VQLA/` : seq_{1,2,3,4,5,6,7,9,10,11,12,14,15,16}. Each sequence folder follows the same folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from EndoVIS18 challange.
            - `vqla`
                - Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                    - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
        - `....`
        - `seq_16`
    - `EndoVis-17-VQLA/` : selected 97 frames from EndoVIS17 challange for external validation. 
        - `left_frames`
        - `vqla`
            - Q&A pairs and bounding box label.
            - `img_features`: Contains img_features extracted from each frame with different patch size.
                - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
    - `featre_extraction/`:
        - `feature_extraction_EndoVis18-VQLA-frcnn.py`: Used to extract features with Fast-RCNN and ResNet101.
        - `feature_extraction_EndoVis18-VQLA-resnet`: Used to extract features with ResNet18 (based on patch size).
- `models/`: 
    - GatedLanguageVisualEmbedding.py : GLVE module for visual and word embeddings and fusion.
    - LViTPrediction.py : our proposed LViT model for VQLA task.
    - VisualBertResMLP.py : VisualBERT ResMLP encoder from Surgical-VQA.
    - visualBertPrediction.py : VisualBert encoder-based model for VQLA task.
    - VisualBertResMLPPrediction.py : VisualBert ResMLP encoder-based model for VQLA task.
- dataloader.py
- train.py
- utils.py

---

## Dataset

[Link] (https://drive.google.com/drive/folders/10kFZFX0RwTosEwUAGNcKKUkb499hGHMj?usp=sharing)
<!-- 1. EndoVis-18-VQA (Image frames can be downloaded directly from EndoVis Challenge Website)
    - [VQLA](https://drive.google.com/file/d/1m7CSNY9PcUoCAUO_DoppDCi_l2L2RiFN/view?usp=sharing)
2. EndoVis-17-VLQA (External Validation Set)
    - [Images & VQLA](https://drive.google.com/file/d/1PQ-SDxwiNXs5nmV7PuBgBUlfaRRQaQAU/view?usp=sharing)   -->

---

## Checkpoint 

[Link] (https://colab.research.google.com/drive/19tRxd0Riaw8D_2Dv12kpJHP9463lF2nW?usp=sharing)
<!-- 1. EndoVis-18-VQA (Image frames can be downloaded directly from EndoVis Challenge Website)
    - [VQLA](https://drive.google.com/file/d/1m7CSNY9PcUoCAUO_DoppDCi_l2L2RiFN/view?usp=sharing)
2. EndoVis-17-VLQA (External Validation Set)
    - [Images & VQLA](https://drive.google.com/file/d/1PQ-SDxwiNXs5nmV7PuBgBUlfaRRQaQAU/view?usp=sharing)   -->

---

### Run training

- Train on EndoVis-18-VLQA 
    ```bash
    python train.py --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver lvit --batch_size 8 --epochs 80
    ```

---

## Evaluation

- Evaluate on both EndoVis-18-VLQA & EndoVis-17-VLQA
    ```bash
    python train.py --validate True --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver lvit --batch_size 8
    ```

---

