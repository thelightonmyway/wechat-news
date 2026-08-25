# wechat-news

“气海无涯”科研资讯微信公众号自动化工具。

```text
QQ Bot
→ 科研新闻 / 论文候选
→ 内容处理
→ 图片与版权检查
→ qihai 微信排版
→ 微信公众号草稿箱
```

> **重要：** 本项目只自动创建微信公众号草稿，不会自动正式发布文章。

## 内容类型

### NEWS

面向科普与科学新闻：从 `config/feeds.yaml` 配置的 RSS / 科研新闻来源获取内容，经过研究主题相关性筛选后形成 NEWS 候选，并可生成中文科普推文。

### PAPER

面向正式发表论文：使用 DOI / OpenAlex 验证并补充 metadata，支持出版商 HTML Figure、可访问的 OA HTML fallback，以及在没有可合法使用的 HTML 图片时使用 PDF + PyMuPDF4LLM 提取 Figure。PDF 可用时还可渲染论文第一页，并生成科研论文解读型推文。

## 核心能力

- RSS feed collection
- 研究主题相关性筛选
- NEWS / PAPER 候选分离
- DOI / OpenAlex metadata 与正式发表验证
- HTML 图片与 Figure 提取
- OA HTML fallback
- PDF Figure 提取（PyMuPDF4LLM fallback）
- 论文第一页渲染
- 图片许可与第三方版权风险过滤
- qihai 微信排版
- 微信公众号草稿创建
- 周一、周三、周五候选定时推送
- QQ Bot 命令控制

## Quick Start

要求 Python 3.10 或更高版本。

```bash
git clone https://github.com/thelightonmyway/wechat-news.git
cd wechat-news
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
```

编辑本地 `.env`，填入需要使用的 QQ、模型、OpenAlex 和微信公众号配置。不要提交 `.env`。

启动与管理：

```bash
./run.sh                 # 前台运行
./run.sh start           # 后台运行
./run.sh status
./run.sh restart
./run.sh stop
```

## QQ Commands

### General

| 命令 | 用途 |
|---|---|
| `/ping` | 检查 Bot 是否在线，返回 `pong` |
| `/status` | 查看候选、推送、模型、OpenAlex、微信与最近错误状态 |
| `/history` | 查看最近 generated / drafted / failed 记录 |

### News

| 命令 | 用途 |
|---|---|
| `/news` | 获取或读取当天 NEWS 候选列表 |
| `/news N` | 查看第 N 个 NEWS 候选详情 |
| `/news N generate` | 使用已配置的文本模型生成第 N 篇 NEWS 稿件 |
| `/news N publish` | 将已生成的第 N 篇 NEWS 稿件排版并创建微信草稿 |

### Paper

| 命令 | 用途 |
|---|---|
| `/papers` | 获取或读取当天 PAPER 候选列表 |
| `/paper N` | 查看第 N 个 PAPER 候选、DOI/OpenAlex 与图片信息 |
| `/paper N generate` | 使用已配置的文本模型生成第 N 篇论文解读 |
| `/paper N publish` | 将已生成的第 N 篇论文解读排版并创建微信草稿 |

`generate` 需要配置 `MODEL_BASE_URL`、`MODEL_API_KEY` 和 `MODEL_NAME`。`publish` 必须在对应文章已经 `generate` 后执行；未配置微信公众号凭据时只完成本地排版 dry-run。

## Documentation

完整文档：<https://wechat-news.readthedocs.io/en/latest/>

## Version

v0.1.0
