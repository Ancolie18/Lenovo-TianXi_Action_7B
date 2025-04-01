import os
import re
import json
import random
import logging
import argparse
import copy
import itertools
from PIL import Image, ImageDraw, ImageColor
from tqdm import tqdm
import time

import torch
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
import torch.multiprocessing as mp


def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).

    Parameters:
        results (list): A list of dictionaries containing sample results.
    
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []

    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)

    return filtered_results

def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }

    # Collect unique values from the results
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))

    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))

    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))

    return combinations

def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    # Calculate text and icon specific metrics using collect_results_to_eval
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics


def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        application=True,
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            application=application,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        application=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        application = combo.get("application")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            application=application,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"app:{application}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        group=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        group = combo.get("group")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            group=group,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"group:{group}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
    
    Parameters:
        results (list): A list of dictionaries containing sample results.
        
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
    
    return metrics


def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    # TODO: comment out function calls based on your need
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)

    # Save detailed results
    result_report["details"] = results

    return result_report



def adjust_resolution(resolution):
    x, y = resolution
    x = x if x % 28 == 0 else x + (28 - x % 28)
    y = y if y % 28 == 0 else y + (28 - y % 28)
    return x, y


def get_input(args, img_size, screenshot_path, user_query, processor):
    x, y = adjust_resolution(img_size)
    qwen_sys_prompt = "You are a helpful assistant."
    user_query_new =f"The image is a screenshot of a computer or mobile phone interface, with a resolution of {x}x{y}. Please provide the coordinates of the object to be operated according to the command, which is as follows: {user_query}.\nYou must output in the following format, and the specific format is as follows: <|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"

    message = [
        {
            "role": "system",
            "content": qwen_sys_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot_path},
                {"type": "text", "text": f"{user_query_new}"}
            ]
        }
    ]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    return text


def extract_output(args, output_text):

    # prompt output format：
    # if skip_special_tokens=False, output_text: <|box_start|>(593,264),(681,354)<|box_end|><|im_end|>
    # if skip_special_tokens=True, output_text: (593,264),(681,354)
    try:
        pattern = r'\((\d+),(\d+)\)'
        matches = re.findall(pattern, output_text)
        bbx_pred = [int(x) for match in matches for x in match]
        if args.verbose:
            logging.info("bbx_pred：%s", bbx_pred)

        if bbx_pred and len(bbx_pred) == 4:
            x1, y1, x2, y2 = bbx_pred
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            point_in_pixel = (center_x, center_y)
            if args.verbose:
                logging.info("point_pred：%s", point_in_pixel)
        else:
            point_in_pixel = None
    except Exception as e:
        point_in_pixel = None
        if args.verbose:
            logging.error("worng_format：%s", e)

    return point_in_pixel


def perform_gui_grounding_troch(args, img_size, screenshot_path, user_query, model, processor):

    text = get_input(args, img_size, screenshot_path, user_query, processor)
    input_image = Image.open(screenshot_path)
    inputs = processor(text=[text], images=[input_image], padding=True, return_tensors="pt").to('cuda')

    # Generate output
    output_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]

    point_pred = extract_output(args, output_text)

    if args.verbose:
        print("======= input text:",text)  
        print("======= Output text: ", output_text)

    return output_text, point_pred 


def eval_sample_positive_gt_qwen2_5(sample, point_pred):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    
    print("\n======= point_pred:", point_pred)
    print("======= bbox_gt:", bbox)
    if point_pred is None or len(point_pred)!=2:
        correctness = "wrong_format"
    elif (bbox[0] <= point_pred[0] <= bbox[2]) and (bbox[1] <= point_pred[1] <= bbox[3]):
        correctness = "correct"
    else:
        correctness = "wrong"
    print("======= correctness:", correctness)
    return correctness


def main(args):

    # loading model
    print("using torch inference...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    logging.info("Model loaded successfully !")

    # load ss-pro tasks
    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")

    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(",")

    if args.language == "all":
        languages = LANGUAGES
    else:
        languages = args.language.split(",")

    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
            task_data = json.load(f)

        # Create the list of tasks to run, one item as an instance. Tasks may be reused.
        for inst_style in inst_styles:  # Expand tasks based on user configurations
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang
                        if lang == "cn":
                            if inst_style!= 'instruction' or gt_type != 'positive':
                                # TODO: Translate the data
                                raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    print(f"Total tasks: {len(tasks_to_run)}")

    # start inference
    results = []    
    corr_action = 0
    num_action = 0
    for sample in tqdm(tasks_to_run):
        num_action += 1
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        user_query = sample["prompt_to_evaluate"]
        img_size = sample["img_size"]
        
        if args.verbose:
            print("--------------------------------------------------------")
            print("sample:",sample)
            print("img_path:",img_path)
            print("user_query:",user_query)
        
        output_text, point_pred = perform_gui_grounding_troch(args, img_size, img_path, user_query, model, processor)
        
        sample_result = {
            "img_path": img_path, 
            "group": sample["group"] if "group" in sample else None,
            "platform": sample["platform"],
            "application": sample["application"],
            "lang": sample["language"],
            "instruction_style": sample["instruction_style"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"], 
            "gt_type": sample["gt_type"],
            "ui_type": sample["ui_type"], 
            "task_filename": sample["task_filename"], 
            "pred": point_pred, 
            "raw_response": output_text
        }
                
        correctness = eval_sample_positive_gt_qwen2_5(sample, point_pred)
        
        if correctness == "correct":
            corr_action += 1
            logging.info("correct, accuracy：%.2f", corr_action / num_action)
        elif correctness == "wrong_format":
            logging.info("wrong_format, accuracy：%.2f", corr_action / num_action)
        else:
            logging.info("wrong, accuracy：%.2f", corr_action / num_action)


        sample_result.update({
            "correctness": correctness,
        })
        results.append(sample_result)

        if args.verbose:
            print(sample_result)
        
    result_report = evaluate(results)

    # Save to file
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    logging.info(f"Evaluation of ScreenSpot finished. saved to {args.log_path}")


GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']


if __name__ == "__main__":
    start_time = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/path/to/your/model/")

    parser.add_argument('--screenspot_imgs', type=str, default="/path/to/ScreenSpot-Pro/images")
    parser.add_argument('--screenspot_test', type=str, default="/path/to/ScreenSpot-Pro/annotations")
    parser.add_argument('--task', type=str, default="all")
    parser.add_argument('--log_path', type=str, default="./eval_results/ss-pro/Qwen2.5-VL-7B-Instruct.json")

    parser.add_argument('--inst_style', type=str, default="instruction", choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, default="en" , choices=LANGUAGES + ['all'], help="Language to use.")
    parser.add_argument('--gt_type', type=str, default="positive", choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")

    parser.add_argument('-v', '--verbose', action='store_true', help="Show input/output of model. By default this is false(enable if set).")
    
    args = parser.parse_args()

    model_name = args.model_path.split("/")[-1]
    args.log_path = "./eval_results/" + model_name +".json"
    logging.info("args：%s", args)

    main(args)

    end_time = time.time()
    duration_min = (end_time - start_time) / 60
    print(f"Inference completed, duration：{duration_min} mins")

