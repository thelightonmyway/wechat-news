项目简介
========

项目用途
--------

“气海无涯” WeChat News 用于把科研资讯处理流程集中到一个本地运行的 QQ Bot：

.. code-block:: text

   QQ Bot
   → 科研新闻 / 论文候选
   → 内容处理
   → 图片与版权检查
   → qihai 微信排版
   → 微信公众号草稿箱

系统保留人工选择步骤。候选、生成和创建草稿均通过 QQ 命令触发；创建草稿后，
仍需在微信公众号后台检查并决定是否正式发布。

两种内容模式
------------

NEWS
~~~~

NEWS 面向科普与科学新闻。系统从 ``config/feeds.yaml`` 中配置的 RSS 和科研新闻
来源获取条目，按研究主题筛选并区分候选，随后可生成中文科普推文。

PAPER
~~~~~

PAPER 面向已经正式发表的论文。系统使用 DOI 和 OpenAlex 验证并补充 metadata，
尝试从出版商 HTML 或可访问的开放 HTML 页面获取正文和 Figure；没有可合法使用的
HTML 图片时，可回退到 PDF 与 PyMuPDF4LLM Figure 提取。PDF 可用时还可渲染论文
第一页，并生成科研论文解读型推文。

核心能力
--------

* RSS feed collection
* 研究主题相关性筛选
* NEWS / PAPER 候选分离
* DOI / OpenAlex metadata 补充与正式发表验证
* HTML 图片与 Figure 提取
* OA HTML fallback
* PDF Figure 提取（PyMuPDF4LLM fallback）
* 论文第一页渲染
* 图片许可与第三方版权风险过滤
* qihai 微信排版
* 微信公众号草稿创建
* 周一、周三、周五候选定时推送
* QQ Bot 命令控制
