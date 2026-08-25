图片与版权
==========

总原则
------

图片发现与图片可发布判定是两个独立步骤。HTML、newspaper4k、Wikimedia Commons、NASA、
OpenAlex 或 PDF 提取只负责提供候选与 metadata；最终是否可用由 ``images/policy.py`` 的
保守规则决定。

News 图片
---------

NEWS 生成流程可根据文章标题相关文本生成英文搜索短语，并依次查询：

* Wikimedia Commons
* Wikimedia Commons 中的 NOAA / NASA 相关结果
* NASA Image and Video Library

候选图片需要通过 metadata 相关性、许可证、尺寸、宽高比和非内容图片过滤。当前 NEWS
正文最多选择 2 张非冗余合规图片，并从中选择封面候选；没有可用图片时继续使用默认封面。

Paper 图片
----------

PAPER 图片来源按当前流程包括：

1. 出版商 HTML 中的 ``figure`` 图片；
2. 出版商 HTML 不可访问时，可访问的 OA HTML mirror（优先 PMC）；
3. 没有可合法使用的 HTML 图片时，正式/参考 PDF + PyMuPDF4LLM Figure fallback；
4. PDF 已下载时渲染论文第一页及微信封面裁剪图。

PDF Figure 只接受与 ``Fig. N`` 图注相邻且尺寸合理的图片区域。补充材料、同行评审文件和
无法确认的图片区域不会作为正式 Figure 使用。当前 PAPER 正文最多选择 4 张非冗余合规图片。
论文第一页可用时会放在正文开头，并优先使用其裁剪图作为微信封面。

允许的许可证
--------------

当前代码允许：

* CC BY
* CC BY-SA
* CC BY-NC
* CC BY-NC-SA
* CC0
* Public Domain

拒绝规则
--------

当前代码拒绝：

* 任意 ND / NoDerivatives 许可证，包括 CC BY-ND 和 CC BY-NC-ND
* unknown 或无法确认可复用的许可证
* all rights reserved、publisher copyright、copyrighted
* 图注或署名中含第三方风险标记：reprinted、reproduced、with permission、third-party
* Google Earth
* Getty、Alamy、Shutterstock

另外，logo、favicon、广告、tracking pixel、banner、sprite，以及尺寸或宽高比明显不适合
正文的图片会在下载阶段被排除。

Metadata 与正文
---------------

每张图片的原始 caption、credit、license、provider、来源 URL 和判定原因保存在文章内部
``metadata.json`` 与 SQLite 图片记录中。正文仅显示脚本生成的简短中文图注，不显示冗长的
license、credit、copyright 或图库说明；这些内部 metadata 应保留用于发布前人工复核。
