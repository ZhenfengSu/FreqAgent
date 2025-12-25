# 1. 首先测试API连接
# python entropy_analyzer.py --test --base-url http://localhost:6001

python entropy_analyzer.py --test-sample

# 2. 运行分析
python entropy_analyzer.py -i trajectories.json -o results.json



python entropy_analyzer.py --diagnose -i trajectories.json