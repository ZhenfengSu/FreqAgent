import concurrent.futures
import os
import argparse
import json
from tqdm import tqdm
import re
import traceback
from openai import OpenAI

# Judge prompts
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
            "prediction": item["prediction"],
            "judgement": response_text
        }

    except Exception as e:
        print(f"Error judgement for question: {question}: {e}")
        return {
            "question": question,
            "answer": correct_answer,
            "prediction": item.get("prediction", ""),
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


def evaluate_file(args):
    """评估单个 JSONL 文件"""
    
    # 加载数据
    print(f"Loading data from: {args.input_file}")
    data = load_jsonl(args.input_file)
    
    if not data:
        print(f"No data found in {args.input_file}")
        return
    
    total_questions = len(data)
    print(f"Total questions: {total_questions}")
    
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )
    
    # 获取 judge prompt
    judge_prompt = get_judge_prompt(args.dataset)
    print(f"Using {args.dataset} judge prompt...")
    
    # 评判所有问题
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                call_llm_judge, 
                item, 
                judge_prompt, 
                client, 
                args.model_name
            ): item for item in data
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(futures), 
            desc="Evaluating"
        ):
            results.append(future.result())
    
    # 统计正确率
    correct_count = 0
    detailed_results = []
    
    for result in results:
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
        
        if is_correct:
            correct_count += 1
        
        detailed_results.append({
            "question": result["question"],
            "answer": result["answer"],
            "prediction": result["prediction"],
            "is_correct": is_correct,
            "judgement": result["judgement"]
        })
    
    # 计算正确率
    accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
    
    # 构建输出结果
    output = {
        "dataset": args.dataset,
        "input_file": args.input_file,
        "statistics": {
            "total_questions": total_questions,
            "correct_count": correct_count,
            "accuracy": round(accuracy, 2)
        },
        "detailed_results": detailed_results
    }
    
    # 保存结果
    input_dir = os.path.dirname(args.input_file)
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = os.path.join(input_dir, f"{input_basename}_results.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Input file: {args.input_file}")
    print("-" * 60)
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions from a single JSONL file"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to JSONL prediction file"
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
    
    if not os.path.isfile(args.input_file):
        print(f"Error: {args.input_file} is not a valid file")
        return
    
    evaluate_file(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"Evaluation Failed: {e}")
        print("Traceback:", error_str)