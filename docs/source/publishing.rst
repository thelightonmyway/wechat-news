微信公众号草稿
==============

发布流程
--------

.. code-block:: text

   generate
   → Markdown 稿件与 metadata
   → qihai formatting
   → 本地/远程图片处理与微信图片上传
   → 微信公众号 draft/add

项目固定调用 vendored ``xiaohu-wechat-format`` 的格式化与发布脚本，并使用 ``qihai`` theme。
如果 ``assets/qihai-header.png`` 存在，排版前会把品牌页眉临时加入 Markdown，发布后还会检查
该页眉是否进入微信草稿内容。

标题与正文
----------

NEWS
~~~~

NEWS 草稿使用候选中文标题（没有中文标题时使用原始标题），并限制在微信标题长度范围内。
正文包含模型生成的科普文章、最多 2 张选中的正文图片以及文章信息。

PAPER
~~~~~

PAPER 草稿标题优先组合“最新成果”、期刊名和文章标题，并遵守微信 64 字符限制。排版时会
移除正文中的一级标题；论文第一页可用时放在正文开头。正文可包含最多 4 张选中的 Figure
或其他合规图片，末尾附论文信息。

封面
----

PAPER 优先使用论文第一页的微信比例裁剪图；否则使用选中的合规封面图片；仍不可用时回退到
``assets/default-cover.jpg``。NEWS 使用选中的合规封面图片或默认封面。

Dry-run 与正式发布
------------------

未配置 ``WECHAT_APP_ID`` 和 ``WECHAT_APP_SECRET`` 时，命令只完成本地 qihai 排版并返回
HTML 路径。配置凭据后，``publish`` 只调用微信公众号草稿接口创建草稿。

.. important::

   项目不会调用微信公众号正式发布接口。草稿必须在微信后台人工检查和发布。
