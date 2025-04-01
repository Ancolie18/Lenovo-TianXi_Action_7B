# lenovo_grounding

This repository provides an evaluation script to run inference on the [ScreenSpot-Pro](https://github.com/zzxslp/ScreenSpot-Pro) benchmark using a custom multimodal model. 
It evaluates the model's performance and logs the results into a JSON file for further analysis.

## 🔍 Project Purpose

This script is designed to:
- Load and run a multimodal model on the ScreenSpot-Pro dataset.
- Evaluate its performance across multiple instruction styles and languages.
- Save evaluation results to a structured log file for easy review.

Supported models include:
- [`grounding7b`](https://huggingface.co/tangliang/grounding7b) (https://huggingface.co/tangliang/grounding7b)
- [`gounding72b`](https://huggingface.co/tangliang/gounding72b) (https://huggingface.co/tangliang/gounding72b)



## 🚀 Usage

### Install Dependencies

Make sure you have all required Python packages installed. It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```


### Run the Evaluation Script
```bash
python run_inference.py \
    --model_path /path/to/your/model \
    --screenspot_imgs /path/to/ScreenSpot-Pro/images \
    --screenspot_test /path/to/ScreenSpot-Pro/annotations \
    --task all \
    --inst_style instruction \
    --language en \
    --gt_type positive \
    --log_path ./eval_results/your_model_name.json \
    --verbose
```

### Script Arguments

| Argument         | Description                                                 | Default                                                                 |
|------------------|-------------------------------------------------------------|-------------------------------------------------------------------------|
| `--model_path`   | Path to the local model checkpoint directory                | `/path/to/your/model/...`                   |
| `--screenspot_imgs` | Path to ScreenSpot-Pro image folder                     | `/path/to/ScreenSpot-Pro/images`                 |
| `--screenspot_test` | Path to annotation JSON files                           | `/path/to/ScreenSpot-Pro/annotations`           |
| `--log_path`     | Path to output evaluation result file                       | `./eval_results/[model_name].json`                                    |
| `--task`         | Task to run (`all`, `vqa`, `caption`, etc.)                 | `all`                                                                  |
| `--inst_style`   | Instruction style to use                                    | `instruction`                                                          |
| `--language`     | Language to use for evaluation                              | `en`                                                                   |
| `--gt_type`      | Ground truth type (`positive`, `negative`)                  | `positive`                                                             |
| `--verbose`      | If set, prints model input/output for each sample           | `False`                                                                |

## 📝 License
This project is licensed under GPL license.
