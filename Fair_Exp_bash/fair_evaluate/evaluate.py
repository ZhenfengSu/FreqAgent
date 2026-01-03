import concurrent.futures
import os
import argparse
import json
import glob
from tqdm import tqdm
import re
import traceback
from openai import OpenAI

# Judge prompts - 你可以根据需要从 prompt.py 导入或直接定义
JUDGE_PROMPT_GAIA = """You are a fair judge. Given a question, a correct answer, and a model's response, determine if the model's response is correct.

Question: {question}
Correct Answer: {correct_answer}
Model's Response: {response}

Is the model's response correct? Please answer with "Correct: Yes" or "Correct: No" and provide a brief explanation."""

JUDGE_PROMPT_BC = """You are a fair judge. Given a question, a correct answer, and a model's response, determine if the model's response is correct.

Question: {question}
Correct Answer: {correct_answer}
Model's Response: {response}

Is the model's response correct? Please answer with "Correct: Yes" or "Correct: No" and provide a brief explanation."""

JUDGE_PROMPT_QA = """You are a fair judge. Given a question, a correct answer, and a model's response, determine if the model's response is correct.

Question: {question}
Correct Answer: {correct_answer}
Model's Response: {response}

Is the model's response correct? Please answer with "Correct: Yes" or "Correct: No" and provide a brief explanation."""


def extract_correct_judgement(response: str) -> str:
    """从 LLM 裁判回复中提取判断结果"""
    match = re.search(r'correct\s*:\s*(yes|no)', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def call_llm_judge(item, judge_prompt, client, model_name):
    """Judge if predicted answer matches ground-truth"""
    try:
        question = item["question"]
        correct_answer = item["answer"]
        response = item["prediction"].strip()

        prompt = judge_prompt.format(
            question=question,
            correct_answer=correct_answer,
            response=response
        )

        max_tries = 10
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = chat_response.choices[0].message.content
                if response_text:
                    break
            except Exception as e:
                if attempt == (max_tries - 1):
                    raise e

        return {
            "question": question,
            "answer": correct_answer,
            "judgement": response_text
        }

    except Exception as e:
        print(f"Error judgement for question: {question}: {e}")
        return {
            "question": question,
            "answer": correct_answer,
            "judgement": "Error",
            "error": str(e)
        }


def load_jsonl(file_path):
    """加载 JSONL 文件"""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def get_sorted_jsonl_files(folder_path):
    """获取文件夹下所有 JSONL 文件并排序"""
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        # 尝试提取文件名中的数字，支持 iter1.jsonl, round_1.jsonl, 1.jsonl 等格式
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[0])
        return 0
    
    jsonl_files.sort(key=extract_number)
    return jsonl_files


def get_judge_prompt(dataset):
    """根据数据集选择合适的 judge prompt"""
    if dataset in ["gaia", "webwalker", "xbench-deepsearch", "hle"]:
        return JUDGE_PROMPT_GAIA
    elif dataset.startswith("browsecomp"):
        return JUDGE_PROMPT_BC
    elif "qa" in dataset:
        return JUDGE_PROMPT_QA
    else:
        return JUDGE_PROMPT_GAIA


def evaluate_folder(args):
    """评估文件夹中的所有 JSONL 文件"""
    
    # 获取所有 JSONL 文件
    jsonl_files = get_sorted_jsonl_files(args.input_folder)
    
    if not jsonl_files:
        print(f"No JSONL files found in {args.input_folder}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {os.path.basename(f)}")
    
    # 加载所有文件的数据
    all_rounds_data = []
    for jsonl_file in jsonl_files:
        data = load_jsonl(jsonl_file)
        all_rounds_data.append({
            "file": jsonl_file,
            "data": data,
            "count": len(data)
        })
        print(f"  {os.path.basename(jsonl_file)}: {len(data)} questions")
    
    # 确定总问题数（以最大数量为准，即完整文件的问题数）
    question_counts = [rd["count"] for rd in all_rounds_data]
    total_questions = max(question_counts)
    print(f"\nTotal unique questions: {total_questions}")
    
    # 计算平均作答次数
    total_answers = sum(question_counts)
    avg_attempts = total_answers / total_questions if total_questions > 0 else 0
    print(f"Total answers across all rounds: {total_answers}")
    print(f"Average attempts per question: {avg_attempts:.2f}")
    
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )
    
    # 获取 judge prompt
    judge_prompt = get_judge_prompt(args.dataset)
    print(f"\nUsing {args.dataset} judge prompt...")
    
    # 对每个 round 进行评判
    all_round_results = []
    
    for round_idx, round_data in enumerate(all_rounds_data):
        round_name = f"round_{round_idx + 1}"
        print(f"\nEvaluating {round_name} ({os.path.basename(round_data['file'])})...")
        
        round_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    call_llm_judge, 
                    item, 
                    judge_prompt, 
                    client, 
                    args.model_name
                ): item for item in round_data["data"]
            }
            
            for future in tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(futures), 
                desc=f"Evaluating {round_name}"
            ):
                round_results.append(future.result())
        
        all_round_results.append({
            "round_name": round_name,
            "file": round_data["file"],
            "count": round_data["count"],
            "results": round_results
        })
    
    # 聚合结果：按问题组织
    question_results = {}
    
    for round_data in all_round_results:
        round_name = round_data["round_name"]
        
        for result in round_data["results"]:
            question = result["question"]
            
            if question not in question_results:
                question_results[question] = {
                    "answer": result["answer"],
                    "rounds": {}
                }
            
            # 判断是否正确
            judgement = result["judgement"]
            is_correct = False
            
            if args.dataset.startswith("browsecomp") or "qa" in args.dataset:
                judge = extract_correct_judgement(judgement)
                if judge and judge.lower() == 'yes':
                    is_correct = True
            else:
                if judgement and "correct" in judgement.lower():
                    judge = extract_correct_judgement(judgement)
                    if judge and judge.lower() == 'yes':
                        is_correct = True
            
            question_results[question]["rounds"][round_name] = {
                "judgement": judgement,
                "is_correct": is_correct
            }
    
    # 计算指标
    
    # 1. 计算每轮的 Pass@1（分母始终为 total_questions）
    round_pass_rates = {}
    for round_data in all_round_results:
        round_name = round_data["round_name"]
        correct_count = 0
        answered_count = round_data["count"]  # 该轮实际回答的问题数
        
        for result in round_data["results"]:
            question = result["question"]
            if question in question_results:
                round_info = question_results[question]["rounds"].get(round_name, {})
                if round_info.get("is_correct", False):
                    correct_count += 1
        
        # pass_rate 以 total_questions 为分母
        pass_rate = (correct_count / total_questions * 100) if total_questions > 0 else 0
        round_pass_rates[round_name] = {
            "correct": correct_count,
            "answered": answered_count,
            "total": total_questions,
            "pass_rate": round(pass_rate, 2)
        }
    
    # 2. Best Pass@1：所有轮次中最高的 Pass@1
    best_pass_at_1 = max(rp["pass_rate"] for rp in round_pass_rates.values()) if round_pass_rates else 0
    best_pass_round = max(round_pass_rates.keys(), key=lambda k: round_pass_rates[k]["pass_rate"]) if round_pass_rates else None
    
    # 3. Any Pass（只要有一轮正确就算正确）
    any_pass_correct = 0
    for question, data in question_results.items():
        if any(r.get("is_correct", False) for r in data["rounds"].values()):
            any_pass_correct += 1
    
    any_pass_rate = (any_pass_correct / total_questions * 100) if total_questions > 0 else 0
    
    # 构建结果
    results = {
        "dataset": args.dataset,
        "input_folder": args.input_folder,
        "files": [os.path.basename(rd["file"]) for rd in all_rounds_data],
        "statistics": {
            "total_questions": total_questions,
            "total_answers": total_answers,
            "num_rounds": len(all_rounds_data),
            "avg_attempts_per_question": round(avg_attempts, 2)
        },
        "metrics": {
            "best_pass_at_1": round(best_pass_at_1, 2),
            "best_pass_round": best_pass_round,
            "any_pass": round(any_pass_rate, 2),
            "any_pass_correct_count": any_pass_correct
        },
        "per_round": round_pass_rates,
        "detailed_results": {
            q: {
                "answer": data["answer"],
                "rounds": {
                    rn: {
                        "is_correct": ri["is_correct"],
                        "judgement": ri["judgement"][:200] + "..." if len(ri["judgement"]) > 200 else ri["judgement"]
                    }
                    for rn, ri in data["rounds"].items()
                }
            }
            for q, data in question_results.items()
        }
    }
    
    # 保存结果
    output_path = os.path.join(args.input_folder, "results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Input folder: {args.input_folder}")
    print(f"Number of files: {len(all_rounds_data)}")
    print("-" * 60)
    print(f"Total questions: {total_questions}")
    print(f"Total answers: {total_answers}")
    print(f"Average attempts per question: {avg_attempts:.2f}")
    print("-" * 60)
    print("Per-round Pass@1 (denominator = total_questions):")
    for round_name, rp in round_pass_rates.items():
        print(f"  {round_name}: {rp['pass_rate']}% ({rp['correct']}/{rp['total']}, answered {rp['answered']})")
    print("-" * 60)
    print(f"Best Pass@1: {best_pass_at_1}% ({best_pass_round})")
    print(f"Any Pass (at least one correct): {any_pass_rate}% ({any_pass_correct}/{total_questions})")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions from multiple JSONL files in a folder"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing JSONL prediction files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaia",
        choices=[
            "gaia", "browsecomp_zh", "browsecomp_zh_small",
            "browsecomp_en", "browsecomp_en_full", "browsecomp_en_small",
            "webwalker", "simple_qa", "simple_qa_small",
            "time_qa", "xbench-deepsearch", "hle"
        ],
        help="Dataset type for selecting appropriate judge prompt"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:6002/v1",
        help="OpenAI API base URL"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="OpenAI API key"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen2.5-72b-instruct",
        help="Model name for LLM judge"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers for evaluation"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a valid directory")
        return
    
    evaluate_folder(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"Evaluation Failed: {e}")
        print("Traceback:", error_str)