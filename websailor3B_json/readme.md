重点检查
- WebSailorKVExtractor
- AgentTrajectoryParser
- KVCacheSpectralAnalyzer

# 逻辑上我们是提取了轨迹，然后对轨迹进行频谱分析，最后进行可视化
# 我不太清楚图片中某块区域是高频低频是如何实现确定的
# 是不是应该把其余区域设置为0，然后去做频谱分析，这样比较合理l