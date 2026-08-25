Scheduler
=========

实际时间表
----------

APScheduler 在 QQ Bot 进程内运行，当前 CronTrigger 为：

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - 星期
     - 默认时间
     - 自动候选类型
   * - 周一
     - 07:00
     - NEWS
   * - 周三
     - 07:00
     - PAPER
   * - 周五
     - 07:00
     - NEWS

实际时间由 ``DAILY_PUSH_TIME`` 控制，时区由 ``DAILY_TIMEZONE`` 控制；默认时区为
``Asia/Shanghai``。

启动补跑
--------

Bot 如果在周一、周三或周五的计划时间之后启动，且当天候选尚未自动推送，会执行一次
startup catch-up。数据库已经记录 ``pushed_at`` 时不会重复自动推送。

自动推送与手动命令
------------------

自动任务只抓取/读取对应内容类型的候选，并把候选列表发送到 ``QQ_TARGET_OPENID``。
未配置目标 OpenID 时，任务记录为 blocked，不发送候选。

手动 ``/news`` 和 ``/papers`` 命令可分别获取对应候选；``generate`` 和 ``publish`` 始终需要
用户显式选择候选。Scheduler 不会自动生成稿件、创建微信草稿或正式发布文章。
