import json
import os
from pathlib import Path

def count_use_round_in_file(file_path):
    """统计单个jsonl文件中use_round的总和"""
    total = 0
    line_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'use_round' in data:
                        total += data['use_round']
                        line_count += 1
                except json.JSONDecodeError as e:
                    print(f"  警告: 跳过无效的JSON行: {e}")
    except Exception as e:
        print(f"  错误: 无法读取文件 {file_path}: {e}")
        return None, None
    
    return total, line_count

def scan_folder(folder_path, recursive=False):
    """扫描文件夹中的所有jsonl文件并统计use_round"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    if not folder.is_dir():
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return
    
    # 查找所有jsonl文件
    if recursive:
        jsonl_files = list(folder.rglob("*.jsonl"))
    else:
        jsonl_files = list(folder.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"在 '{folder_path}' 中没有找到jsonl文件")
        return
    
    print(f"在 '{folder_path}' 中找到 {len(jsonl_files)} 个jsonl文件\n")
    print("=" * 70)
    
    grand_total = 0
    grand_line_count = 0
    results = []
    
    for file_path in sorted(jsonl_files):
        total, line_count = count_use_round_in_file(file_path)
        
        if total is not None:
            results.append({
                'file': file_path.name,
                'path': str(file_path),
                'total': total,
                'count': line_count
            })
            grand_total += total
            grand_line_count += line_count
            
            print(f"文件: {file_path.name}")
            print(f"  路径: {file_path}")
            print(f"  包含use_round的记录数: {line_count}")
            print(f"  use_round总和: {total}")
            if line_count > 0:
                print(f"  平均值: {total / line_count:.2f}")
            print("-" * 70)
    
    # 输出汇总信息
    print("\n" + "=" * 70)
    print("汇总统计:")
    print(f"  总文件数: {len(results)}")
    print(f"  总记录数: {grand_line_count}")
    print(f"  use_round总和: {grand_total}")
    if grand_line_count > 0:
        print(f"  整体平均值: {grand_total / grand_line_count:.2f}")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='统计jsonl文件中use_round的总和')
    parser.add_argument('folder', help='要扫描的文件夹路径')
    parser.add_argument('-r', '--recursive', action='store_true', 
                        help='是否递归扫描子文件夹')
    
    args = parser.parse_args()
    
    scan_folder(args.folder, args.recursive)