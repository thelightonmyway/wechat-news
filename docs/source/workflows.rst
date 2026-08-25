内容工作流
==========

News workflow
-------------

.. code-block:: text

   配置的 RSS / 科研新闻来源
   → 48 小时抓取（候选不足时扩展到 168 小时）
   → 研究主题筛选与去重
   → NEWS 候选
   → 选择候选并 generate
   → 公共图片 metadata 搜索与版权过滤
   → 下载合规图片并生成中文图注
   → qihai 排版
   → 微信公众号草稿

NEWS 候选按 DOI、canonical URL、规范化标题和相似标题去重。候选会先经过确定性评分，
文本模型配置可用时再完成候选标题处理和选择；模型不可用时保留确定性回退。

Paper workflow
--------------

.. code-block:: text

   PAPER 候选
   → DOI / OpenAlex 正式发表验证与 metadata
   → 出版商 HTML 正文与 Figure
   → 访问失败时尝试 OA HTML mirror
   → 没有可合法使用的 HTML 图片时尝试 PDF + PyMuPDF4LLM Figure
   → PDF 可用时渲染论文第一页
   → generate 论文解读
   → qihai 排版
   → 微信公众号草稿

PAPER 模式要求 OpenAlex 返回 DOI、期刊、正式发表日期和受支持的论文类型。生成时优先使用
论文 metadata、abstract 和论文正文；新闻正文仅作为补充。PDF Figure fallback 只在没有
可合法使用的 HTML 图片时触发，并过滤补充材料、同行评审文件等非正文 PDF。

候选与手动命令
--------------

Scheduler 负责按时间向 QQ 推送候选列表。用户仍需通过 ``/news``、``/papers``、
``/news N ...`` 或 ``/paper N ...`` 选择、生成和创建草稿。自动候选推送不会自动生成文章，
也不会自动创建或正式发布微信文章。
