配置
====

配置来源
--------

项目从仓库根目录 ``.env`` 读取配置。变量模板位于 ``.env.example``；模板只包含变量名、
空占位符和安全默认值。不要把真实 ``.env``、token、OpenID 或 secret 提交到 Git。

QQ
--

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - 变量
     - 必需性
     - 作用
   * - ``QQ_APP_ID``
     - 必需
     - QQ Bot App ID
   * - ``QQ_CLIENT_SECRET``
     - 必需
     - QQ Bot Client Secret
   * - ``QQ_TARGET_OPENID``
     - 可选
     - 定时候选推送目标；为空时，Bot 会尝试绑定首个私聊用户

文本模型
--------

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - 变量
     - 必需性
     - 作用
   * - ``MODEL_BASE_URL``
     - 可选
     - OpenAI-compatible API 基础 URL
   * - ``MODEL_API_KEY``
     - 可选
     - 文本模型 API key
   * - ``MODEL_NAME``
     - 可选
     - 模型名称

三项同时配置后，模型功能才会启用。模型用于候选标题/筛选、图片搜索关键词、中文图注和
文章生成；未配置时，相关步骤会拒绝执行或使用代码中的确定性回退。

OpenAlex
--------

``OPENALEX_API_KEY``
   OpenAlex API key。PAPER 候选的正式发表验证和 DOI metadata 补充依赖该配置。

微信公众号
----------

``WECHAT_APP_ID``
   微信公众号 App ID。

``WECHAT_APP_SECRET``
   微信公众号 App Secret。

``WECHAT_AUTHOR``
   草稿作者字段与 qihai 页眉作者标签。

未同时配置微信 App ID 与 App Secret 时，``publish`` 只执行本地排版 dry-run，不会创建草稿。

Scheduler
---------

``DAILY_PUSH_TIME``
   周一、周三、周五候选推送时间，格式为 ``HH:MM``；默认 ``07:00``。

``DAILY_TIMEZONE``
   调度时区；默认 ``Asia/Shanghai``。

本地状态
--------

SQLite 数据库固定写入 ``data/news.db``。生成稿件、图片、PDF、Figure、日志和临时文件分别
位于 ``articles/``、``data/``、``logs/``、``tmp/`` 等运行时目录，均不进入公开仓库。
