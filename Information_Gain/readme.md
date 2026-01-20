[
  {
    "question": "问题内容",
    "ground_truth_answer": "标准答案",
    "rollout_id": "xxx",
    "total_rounds": 5,
    "final_prediction": "模型最终预测",
    "termination": "answer",
    "rounds": [
      {
        "round_id": 1,
        "sampled_answers": ["答案1", "答案2", ...],
        "answer_clusters": {"答案A": 15, "答案B": 5},
        "num_clusters": 2,
        "entropy": 0.8113,
        "information_gain": 0.0
      },
      {
        "round_id": 2,
        "sampled_answers": [...],
        "answer_clusters": {"答案A": 18, "答案C": 2},
        "num_clusters": 2,
        "entropy": 0.4690,
        "information_gain": 0.3423
      }
    ],
    "summary": {
      "initial_entropy": 0.8113,
      "final_entropy": 0.0,
      "total_ig": 0.8113,
      "avg_ig_per_round": 0.2028,
      "max_ig_round": 3,
      "entropy_trajectory": [0.8113, 0.4690, 0.2516, 0.0],
      "ig_trajectory": [0.0, 0.3423, 0.2174, 0.2516]
    }
  }
]