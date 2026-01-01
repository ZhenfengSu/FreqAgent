主要修改说明
新增功能
plot_entropy_curve() 方法: 绘制单个 case 的 token 熵变化曲线，支持：

X 轴为 token 位置
不同颜色的背景区域标注 phase 范围（think, tool_call, tool_response, answer）
显示整体统计信息和各 phase 的统计信息
可选择是否显示 token 文本
analyze_single_case() 方法: 分析单个 case，返回详细的 token、熵值和 phase 范围信息

analyze_and_plot_case() 方法: 组合分析和绘图

interactive_case_viewer() 方法: 交互式逐个查看 case

新增命令行参数
--plot / -p: 绘制单个 case 的熵曲线
--case / -c: 指定要分析的 case 索引（默认为 0）
--save-plot: 保存图片路径
--interactive: 交互式查看模式
--show-tokens: 在图表中显示 token 文本
--figsize: 自定义图片大小