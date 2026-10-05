# Error Recognition in Procedural Videos using Generalized Task Graph

- [Preparation](#Preparation)
- [Training](#Training)
- [Inference](#Inference)


This is the official implementation of [Error Recognition in Procedural Videos using Generalized Task Graph](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Error_Recognition_in_Procedural_Videos_using_Generalized_Task_Graph_ICCV_2025_paper.pdf)

Please cite our ICCV 2025 paper if our paper/implementation is helpful for your research:
```
@InProceedings{Lee_2025_ICCV,
    author    = {Lee, Shih-Po and Elhamifar, Ehsan},
    title     = {Error Recognition in Procedural Videos using Generalized Task Graph},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {10009-10021}
}
```

## Preparation

### Setup the conda environment.

Adjust the torch and CUDA version according to your hardware

```
conda env create -f environment.yml
```

### Download Datasets
To download the original EgoPER dataset, please visit [EgoPER](https://github.com/robert80203/EgoPER_official)

To download the original CaptainCook4D dataset, please visit [CaptainCook4D](https://captaincook4d.github.io/captain-cook/)


**To download the processed datsets (including annotations, pre-extracted features, training/test splits) and pre-trained weights of GTG2Vid for both EgoPER and CaptainCook4D**, please send a request to lee.shih@northeastern.edu with the following information:

- Your Full Name
- Institution/Organization
- Advisor/Supervisor Name
- Current Position/Title
- Emaill Address (with institutional domain name)
- Purpose (e.g., download the dataset or pre-trained weight or both, for research purpose or others)

Create a data/ folder with the following structure and move data/labels accordingly.

```
- data
    - EgoPER
        - action2idx.json
        - idx2action.json
        - actiontype2idx.json
        - idx2actiontypes.json
        - coffee/
            - vc_v_features_10fps/
            - refined_label_v3/
            - vc_chatgpt4omini_error_features/
            - vc_normal_action_features/
            - chatgpt4omini_error.txt
            - normal_actions.txt
            - training.txt
            - validation.txt
            - test.txt
        - oatmeal/
        - pinwheels/
        - tea/
        - quesadilla/
    - CaptainCook4D
        - action2idx.json
        - idx2action.json
        - actiontype2idx.json
        - idx2actiontypes.json
        - breakfastburritos/
            - vc_v_features_10fps/
            - labels_10fps/
            - vc_chatgpt4omini_error_features/
            - vc_normal_action_features/
            - chatgpt4omini_error.txt
            - normal_actions.txt
            - training.txt
            - test.txt
        - cucumberraita/
        - microwaveeggsandwich/
        - ramen/
        - spicedhotchocolate/
```

### Update configuration
Please update the configuration file accordingly (e.g., configs/EgoPER/tea/vc_4omini_post_db0.6.json)
- For EgoPER, set the value of key 'root_data_dir' to data/EgoPER
- For CaptainCook4D, set the value of key 'root_data_dir' to data/CaptainCook4D 

## Training
- EgoPER
```
./train_EgoPER.sh
```

- CaptainCook4D
```
./train_EgoPER.sh
```

## Evaluation
- Specify the ```--dir```
    - e.g., 
    ```
    python main.py --config configs/EgoPER/tea/vc_4omini_post_db0.6.json --dir best --eval --vis
    ```
- EgoPER
```
./eval_EgoPER.sh
```

**Note that coffee will take longer time to process as it has a large task graph.**

- CaptainCook4D
```
./eval_EgoPER.sh
```

**Note that cucumberraita will take longer time to process as it has a large task graph.**

### Check out the results
For example
```
- ckpts
    - EgoPER
        - tea
            - best
                - log
                    - action_segmentation.txt
                    - error_detection.txt
                    - error_recognition.txt
```

Find your desired metrics and results in each .txt file.

For your informationm, you can find the task graphs for the tasks in both EgoPER and CaptainCook in ```datasets/loader_graph.py```