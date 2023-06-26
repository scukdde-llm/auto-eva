## LLM Documents Question Answering 自动评测框架

* splitter  : 用于将 document 切分为不同文档片段
* irsystem  : 文档检索系统，根据查询语句检索出文档片段
* predictor : 大语言模型调用接口
* rating    : 使用模型评价结果

配置由 `auto_eva.toml` 文件决定  
测试数据文件和结果文件为 `json` 格式  
```json
[{"q":"question", "a": "answer"}]
```