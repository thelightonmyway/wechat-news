Scheduler
=========

Default schedule
----------------

APScheduler runs inside the QQ Bot process. The current CronTrigger uses the configured time on
Monday, Wednesday, and Friday:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Day
     - Default time
     - Candidate type
   * - Monday
     - 07:00
     - NEWS
   * - Wednesday
     - 07:00
     - PAPER
   * - Friday
     - 07:00
     - NEWS

``DAILY_PUSH_TIME`` controls the time and ``DAILY_TIMEZONE`` controls the timezone. Defaults are
``07:00`` and ``Asia/Shanghai``.

Startup catch-up
----------------

If the Bot starts after the configured time on Monday, Wednesday, or Friday and the candidate list
has not already been delivered that day, it runs one startup catch-up. A stored ``pushed_at`` value
prevents duplicate automatic delivery.

Delivery target
---------------

Scheduled candidate lists are sent to ``QQ_TARGET_OPENID``. If it is empty, the first private-
message sender is bound automatically. A scheduled run with no target records a blocked state and
does not send the list.

Automatic versus manual actions
-------------------------------

The scheduler only retrieves or reuses the appropriate candidate list and sends it through QQ.
The user must still select a candidate and explicitly run ``generate`` and ``publish``. No scheduled
job automatically creates a WeChat draft or publishes a final post.
