# KBAlign: Efficient Self Adaptation on Specific Knowledge Bases

This is the source code for [KBAlign: Efficient Self Adaptation on Specific Knowledge Bases](https://arxiv). Currently, the code comments are not perfect, and we will continue to supplement them.

## Introduction
<div align="center">
<img src=https://github.com/THUNLP/KBAlign/blob/main/figs/KBAlign-main.png width=60% />
</div>

We propose an approach designed for efficient adaptation to downstream tasks involving knowledge bases (KBs), trying to unleash the self-learning potential of LLMs. Experiments on different datasets & backbone models demonstrate the general effectiveness of KBAlign.

<div align="center">
<img src=https://github.com/THUNLP/KBAlign/blob/main/figs/KBAlign-result.png width=100% />
</div>


### Code Overview

The use for files are briefly described below:

- `requirements.txt`: environmental setup file
- `iter_pipeline(_llama).sh`: iterative tuning bash file
- `self_annotation_data/`
  - `long_pipeline.sh`: long-dependency annotation bash file
  - `short_pipeline.sh`: short-dependency annotation bash file
  - `merge.sh`: merge annotated data or mix general domain data
  - `merge.py`: merge data python file 
  - `long_dependency/`: long-dependency annotation code directory 
  - `short_dependency/`: short-dependency annotation code directory 
- `finetune`
  - `LLaMA-Factory`: LLaMA Factory finetune directory
  - `mc_finetune`: Model Center finetune directory
  - `bmtrainMiniCPM_hugMiniCPM.py`: model format conversion between bmtrainMiniCPM and hugMiniCPM
  - `hugMiniCPM_bmtrainMiniCPM.py`: model format conversion between hugMiniCPM and bmtrainMiniCPM
  - `change_info.py`: updates the dataset for LLaMA Factory
- `verify`
  - `split_verify.py`: splits annotated data
  - `gen_iterate_verify.py`: generates self-verified data
- `eval`
  - `eval.sh`: evaluation bash file
  - `eval.py`: evaluation python file
  - `config.json`: evaluation config file
- `utils/`: utils directory
- `prompt`
  - `eval.json`: prompt for evaluation
  - `jec_eval.json`: prompt for jec dataset evaluation
  - `short_annotation.json`: prompt for short-dependency annotation


## Requirement

The model is implemented using PyTorch. We strongly recommend you to manage your environment using Anaconda, since we switch the environment requirement for the LLaMA-based pipeline. 

The versions of main packages used are shown below.

- numpy==1.24.4
- torch==2.4.0
- transformers==4.44.0
- model-center==1.0.2
- bmtrain==0.2.3
- vllm==0.5.4
  
To set up the dependencies, you can run the following command:
```
pip install -r requirements.txt
```

As for LLaMA-based pipeline, you can create a new conda environment and run the following command:
```
pip install -r ./finetune/LLaMA-Factory/requirements.txt
```

## Data Annotation

- First, save your raw knowledge materials into json file, in the format of: [Homogeneous data(str), Homogeneous data(str),...] 
A segment of LooGLE raw data is saved in `data/loogle/KB/KB_0.json`, and you can use this file to try the annotation pipeline.
Homogeneous data means knowledge on the same topic, content from the same document, or even segments of text related to the same subject area. The segmentation does not enforce a strict length requirement; it primarily depends on logical grouping and relevance within the data.

- Then, change the related arguments in `./self_annotation_data/short_pipeline.sh`, including the input/output file path, the annotation backbone model type and path, language, chunk size, and so on.

- Run `bash ./self_annotation_data/short_pipeline.sh` to conduct the short-denpendency data annotation. Correspondingly, run `bash ./self_annotation_data/long_pipeline.sh` for the long-denpendency annotation.

- You can check `--ChatGLM3_data_format_dir` and `--bm_data_format_dir` to see the self-annotation results. Notice that you can also try annotating data with higher quality, with openai API keys. 

## Iterative Tuning

- With the annotated data, you can conduct the tuning pocess. First, modify the needed arguments in `iter_pipeline.sh`.

- If you are trying to tune the llama-based model, go to `iter_pipeline_llama.sh`. You can also modify the LoRA settings in `./finetune/LLaMA-Factory/config/llama3_lora_sft.yaml` and `./LLaMA-Factory/config/llama3_merge_lora.yaml`. You can modify the dataset settings in `./finetune/LLaMA-Factory/data/dataset_info.json`

- Now, run `bash iter_pipeline.sh` or `bash iter_pipeline_llama.sh` to conduct the tuning process.

## Evaluation

- Please go to `eval/` for evaluation. You can save the data items that you want to test under `data/...`. Modify the related settings in `config.json`, and then run `bash eval.bash`.

- You can also download our tuned checkpoints to directly test the performance.

## Key Arguments for Modification

Below is a table outlining the key arguments that require modification. For more detailed explanations and additional parameters, refer to the bash files and search for todo.


| **Parameter Name**          | **File Name(s)**                  | **Explanation/Example**                                                    |
|------------------------------|------------------------------------|-----------------------------------------------------------------------------|
| `bm_data_format_dir`        | `iter_pipeline.sh`, `long_pipeline.sh`, `short_pipeline.sh` | Directory path for bm data formatting. Example: `data/bm/YOUR_PATH`        |
| `ChatGLM3_data_format_dir`  | `iter_pipeline_llama.sh`, `long_pipeline.sh`, `short_pipeline.sh` | Path for ChatGLM3 formatted data. Example: `data/ChatGLM3/YOUR_PATH`       |
| `Finetune_file_name`        | `iter_pipeline.sh`, `iter_pipeline_llama.sh` | Name of the fine-tuning file. Example: `YOUR_FINETUNE_FILE_NAME`            |
| `model_name_or_path`        | `iter_pipeline.sh`, `short_pipeline.sh`      | Path to the pre-trained model. Example: `YOUR_PATH_TO_LLM`                  |
| `model_save_path`           | `iter_pipeline.sh`, `iter_pipeline_llama.sh` | Path to save the fine-tuned model. Example: `$SCRIPT_DIR/model/finetuned_model` |
| `kb_path`                   | `iter_pipeline.sh`, `iter_pipeline_llama.sh`, `eval.sh`, `long_pipeline.sh` | Knowledge base JSON file path. Example: `data/loogle/KB/KB_{id}.json`       |
| `kb_emb_path`               | `iter_pipeline.sh`, `iter_pipeline_llama.sh`, `eval.sh` | Knowledge base embeddings path. Example: `data/loogle/embeddings/{bge_type}/sentence_embeddings_{id}_{chunk_size}.pkl` |
| `kb_sentence_path`          | `iter_pipeline.sh`, `iter_pipeline_llama.sh`, `eval.sh` | Path to KB sentences. Example: `data/loogle/sentences/sentence_sentences_{id}_{chunk_size}.pkl` |
| `iter_num`                  | `iter_pipeline.sh`, `iter_pipeline_llama.sh` | Number of iterations. Example: `1`                                         |
| `train_step_num`            | `iter_pipeline.sh`, `iter_pipeline_llama.sh` | Number of training steps per iteration. Example: `1`                        |
| `split_num`                 | `iter_pipeline.sh`, `iter_pipeline_llama.sh` | Number of splits for processing. Example: `1`                               |
| `annotate_path`             | `long_pipeline.sh`                | Path for KB annotation. Example: `data/loogle/KB/KB_annotate.json`          |
| `sentences_path`            | `long_pipeline.sh`                | Path for sentence data. Example: `data/loogle/sentences/sentence_sentences_0_128.pkl` |
| `sentences_emb_path`        | `long_pipeline.sh`                | Path for sentence embeddings. Example: `data/loogle/embeddings/BAAI/bge-large-en-v1.5/sentence_embeddings_0_128.pkl` |
| `documents_path`            | `long_pipeline.sh`                | Path for cached documents. Example: `cache/loogle_documents_path.pkl`       |
| `bge_type`                  | `long_pipeline.sh`, `short_pipeline.sh` | Type of embedding model. Example: `BAAI/bge-large-en-v1.5`                  |
| `language`                  | `short_pipeline.sh`               | Language for processing. Example: `English`                                 |
| `model_type`                | `short_pipeline.sh`               | Type of model used. Example: `llama`, `cpm`, or `gpt`                       |
| `tokenizer_path`            | `short_pipeline.sh`               | Path to tokenizer files. Example: `YOUR_PATH_TO_TOKENIZE`                   |
| `cache_dir`                 | `short_pipeline.sh`, `eval.sh`    | Directory for caching. Example: `YOUR_HUGGING_FACE_CACHE_DIR`               |
| `qa_model_path`             | `eval.sh`                         | Path to the fine-tuned QA model. Example: `YOUR_FINETUNE_MODEL_PATH`        |


## Citation

Please cite our work if it helps.
```
@article{zeng2024kbalign,
  title   = {KBAlign: Efficient Self Adaptation on Specific Knowledge Bases},
  author  = {Zheni, Zeng and Yuxuan, Chen and Shi, Yu and Yukun, Yan and Zhenghao, Liu and Shuo, Wang and Xu, Han and Zhiyuan, Liu and Maosong, Sun},
  year    = {2024},
  journal = {arXiv preprint arXiv:2411.14790}
}
```
