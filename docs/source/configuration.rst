Configuration
=============

Configuration file
------------------

wechat-news reads configuration from ``.env`` in the repository root. Create it from the public
template and restrict its permissions:

.. code-block:: bash

   cp .env.example .env
   chmod 600 .env

Safe complete example
---------------------

Replace every placeholder with credentials from your own accounts. If you do not use an
optional integration, clear its value instead of leaving a placeholder that looks configured.

.. code-block:: ini

   QQ_APP_ID=YOUR_QQ_APP_ID
   QQ_CLIENT_SECRET=YOUR_QQ_APP_SECRET
   QQ_TARGET_OPENID=

   OPENALEX_API_KEY=YOUR_OPENALEX_API_KEY

   MODEL_BASE_URL=https://api.example.com/v1
   MODEL_API_KEY=YOUR_MODEL_API_KEY
   MODEL_NAME=YOUR_MODEL_NAME

   WECHAT_APP_ID=YOUR_WECHAT_APP_ID
   WECHAT_APP_SECRET=YOUR_WECHAT_APP_SECRET
   WECHAT_AUTHOR=YOUR_AUTHOR_NAME

   DAILY_PUSH_TIME=07:00
   DAILY_TIMEZONE=Asia/Shanghai

QQ settings
-----------

``QQ_APP_ID``
   AppID for your own QQ Bot application. Required.

``QQ_CLIENT_SECRET``
   AppSecret or ClientSecret for your own QQ Bot application. Required.

``QQ_TARGET_OPENID``
   Recipient for scheduled candidate delivery. When empty, the first user who sends the Bot a
   private message is atomically bound and written to the local ``.env``. An existing non-empty
   value is never overwritten automatically.

Model settings
--------------

``MODEL_BASE_URL``
   Base URL of your OpenAI-compatible API endpoint.

``MODEL_API_KEY``
   API key for that endpoint.

``MODEL_NAME``
   Model identifier accepted by that endpoint.

All three values must be set for model features to be enabled.

OpenAlex setting
----------------

``OPENALEX_API_KEY``
   Optional for NEWS-only use, but required for the current PAPER discovery and formal-publication
   verification workflow. It is recommended for a full deployment.

WeChat settings
---------------

``WECHAT_APP_ID``
   AppID from your own WeChat Official Account.

``WECHAT_APP_SECRET``
   AppSecret from your own WeChat Official Account.

``WECHAT_AUTHOR``
   Author name passed to WeChat draft creation and displayed by the included formatting setup.

Both AppID and AppSecret must be set for WeChat integration to be considered configured. Without
them, ``publish`` performs a local formatting dry-run instead of creating a draft.

Scheduler settings
------------------

``DAILY_PUSH_TIME``
   Candidate delivery time in ``HH:MM`` format. Default: ``07:00``.

``DAILY_TIMEZONE``
   IANA timezone used by the scheduler. Default: ``Asia/Shanghai``.

Local data
----------

The SQLite database is stored at ``data/news.db``. Generated articles, downloaded images, PDFs,
figures, logs, and temporary files are written under ignored runtime directories such as
``articles/``, ``data/``, ``logs/``, and ``tmp/``.

Credential safety
-----------------

Never commit ``.env`` or the runtime-generated ``vendor/xiaohu-wechat-format/config.json``.
Do not paste AppSecrets, API keys, tokens, or OpenIDs into GitHub Issues, screenshots, or public
logs. Rotate a credential immediately through its provider if it is exposed.
