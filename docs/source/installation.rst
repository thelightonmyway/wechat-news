安装与运行
==========

环境要求
--------

* Python 3.10 或更高版本
* ``pip``
* Git
* 可访问 QQ Bot、RSS、OpenAlex、模型和微信接口的网络环境（按实际启用功能）

安装
----

.. code-block:: bash

   git clone https://github.com/thelightonmyway/wechat-news.git
   cd wechat-news
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   cp .env.example .env
   chmod 600 .env

编辑 ``.env`` 后再启动。真实凭据只能保存在本地 ``.env`` 中。

运行
----

前台运行：

.. code-block:: bash

   ./run.sh

后台进程管理：

.. code-block:: bash

   ./run.sh start
   ./run.sh status
   ./run.sh restart
   ./run.sh stop

``run.sh`` 优先使用项目内 ``.venv/bin/python``。后台模式使用 supervisor
循环，在 Bot 子进程异常退出后按退避间隔重新启动。日志和 PID 文件写入 ``logs/``，
该目录不进入 Git。
