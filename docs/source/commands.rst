QQ 命令
========

命令来自当前 ``bot/commands.py`` 和 ``bot/bridge.py`` 实现。

General
-------

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - 命令
     - 用途
     - 示例
   * - ``/ping``
     - 检查 Bot 是否在线，返回 ``pong``
     - ``/ping``
   * - ``/status``
     - 查看候选、推送、模型、OpenAlex、微信和最近错误状态
     - ``/status``
   * - ``/history``
     - 查看最近 10 条 generated / drafted / failed 记录
     - ``/history``

News
----

.. list-table::
   :header-rows: 1
   :widths: 30 42 28

   * - 命令
     - 用途
     - 示例
   * - ``/news``
     - 获取或读取当天 NEWS 候选列表
     - ``/news``
   * - ``/news N``
     - 查看第 N 个 NEWS 候选详情
     - ``/news 2``
   * - ``/news N generate``
     - 使用文本模型生成第 N 篇 NEWS 稿件
     - ``/news 2 generate``
   * - ``/news N publish``
     - 排版已生成稿件并创建微信公众号草稿
     - ``/news 2 publish``

Paper
-----

.. list-table::
   :header-rows: 1
   :widths: 30 42 28

   * - 命令
     - 用途
     - 示例
   * - ``/papers``
     - 获取或读取当天 PAPER 候选列表
     - ``/papers``
   * - ``/paper N``
     - 查看第 N 个 PAPER 候选、DOI/OpenAlex 和图片信息
     - ``/paper 1``
   * - ``/paper N generate``
     - 使用文本模型生成第 N 篇论文解读
     - ``/paper 1 generate``
   * - ``/paper N publish``
     - 排版已生成的论文解读并创建微信公众号草稿
     - ``/paper 1 publish``

执行规则
--------

* ``N`` 是当前候选列表中的序号。
* ``generate`` 需要同时配置 ``MODEL_BASE_URL``、``MODEL_API_KEY`` 和 ``MODEL_NAME``。
* ``publish`` 不会隐式调用 ``generate``；必须先生成对应文章。
* 未配置微信公众号凭据时，``publish`` 返回本地排版 dry-run 结果。
* 所有成功的微信操作都只创建草稿，不会正式发布。
