# Open Source Reuse

本项目保留当前业务代码与 ``vendor/xiaohu-wechat-format`` 的集成方式。下表记录直接使用或随项目 vendor 的主要开源项目；具体许可证以所安装版本的包 metadata 和上游许可证文件为准。

| 用途 | 项目/服务 | 集成方式 | 许可证或使用说明 |
|---|---|---|---|
| RSS 解析 | feedparser | Python 依赖 | BSD-2-Clause |
| 正文抽取 | trafilatura | Python 依赖 | Apache-2.0 |
| 新闻 metadata / 图片发现 | newspaper4k | Python 依赖 | MIT |
| HTML 解析 | Beautiful Soup | Python 依赖 | MIT |
| HTTP 客户端 | aiohttp / HTTPX | Python 依赖 | Apache-2.0 / BSD-3-Clause |
| 论文 metadata | PyAlex / OpenAlex | Python 依赖与 OpenAlex API | PyAlex 为 MIT；OpenAlex 数据遵循其服务条款 |
| 定时任务 | APScheduler | Python 依赖 | MIT |
| PDF 渲染与提取 | PyMuPDF / PyMuPDF4LLM | Python 依赖 | AGPL-3.0 或商业许可；部署与分发时应核对适用条款 |
| 文本模型客户端 | OpenAI Python client | OpenAI-compatible API 客户端 | Apache-2.0 |
| Markdown 处理 | Python-Markdown | Python 依赖 | BSD-3-Clause |
| 微信排版与草稿 | xiaohu-wechat-format | 保存在 ``vendor/xiaohu-wechat-format``；包含本项目使用的 qihai theme 与适配 | 上游 README 声明 MIT；保留上游 README 与 attribution |
| 公共图片发现 | Wikimedia Commons | 按单张图片 metadata 与许可证判定 | 每张图片许可证不同，必须逐项保留内部 license/credit metadata |
| 公共图片发现 | NASA Image and Video Library | API / 图片 metadata | 依 NASA 媒体使用指南与单项 copyright metadata 判定，不自动视为公共领域 |

## Vendored component

``vendor/xiaohu-wechat-format`` 继续作为普通 vendored source 使用，不改为 Git submodule。项目不会提交该工具运行时生成的 ``config.json``；真实微信公众号凭据只保存在本地配置中。
