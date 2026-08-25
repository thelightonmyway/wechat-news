故障排查
========

QQ Bot 没有响应
---------------

* 运行 ``./run.sh status`` 检查 supervisor 与 Bot 子进程。
* 检查 ``logs/bot.log`` 和 ``logs/supervisor.log``。
* 确认 ``QQ_APP_ID``、``QQ_CLIENT_SECRET`` 正确，且网络可以访问 QQ Bot gateway。
* ``/ping`` 应返回 ``pong``。

模型功能不可用
--------------

``generate`` 提示模型未配置时，检查 ``MODEL_BASE_URL``、``MODEL_API_KEY`` 和
``MODEL_NAME`` 是否同时设置。候选标题处理失败时，系统可能回退到确定性候选，但文章生成
不会在缺少模型配置时继续。

OpenAlex timeout 或 PAPER 候选为空
---------------------------------------

* 检查 ``OPENALEX_API_KEY``。
* OpenAlex 超时与 429/5xx 会进行有限次数重试；外部服务持续不可用时稍后再试。
* PAPER 候选还必须有 DOI、期刊、有效发表日期、受支持的论文类型，并通过研究主题筛选。

出版商 HTML 返回 403
---------------------

PAPER 流程会在出版商 HTML 访问失败时尝试 metadata 中的 OA HTML mirror。若仍不可访问，且能
找到正式/参考 PDF，则生成阶段可尝试 PDF Figure fallback。外部站点临时失败时可稍后重试。

论文图片不可用
--------------

图片可能因许可证未知、ND、all rights reserved、第三方版权标记、下载失败或不符合尺寸要求而
被拒绝。检查文章目录中的 ``metadata.json`` 和日志中的判定原因；不要手工绕过版权策略。

微信公众号草稿失败
-------------------

* 确认 ``WECHAT_APP_ID``、``WECHAT_APP_SECRET`` 和 ``WECHAT_AUTHOR`` 配置正确。
* 微信错误 40164 通常表示当前公网 IP 未加入微信公众号后台 IP 白名单。
* 先确认对应文章已经执行 ``generate``。

缺少 qihai 页眉或封面
---------------------

* 品牌页眉应位于 ``assets/qihai-header.png``。
* 默认封面应位于 ``assets/default-cover.jpg``。
* PAPER 的论文第一页封面只有在 PDF 已成功下载并渲染时可用；否则回退到选中图片或默认封面。

外部 API 临时失败
------------------

RSS、图片站点、OpenAlex、模型、QQ 和微信均为外部服务。查看日志中的 HTTP 状态与错误类型，
确认代理和网络设置后再重试；不要把日志、下载文件或临时调试输出提交到 Git。
