Installation
============

Prerequisites
-------------

Before installing the application, prepare:

* Python 3.10 or newer and ``pip``;
* Git;
* a machine or server that can remain running when you want the QQ Bot and scheduler available;
* your own QQ Bot credentials;
* your own WeChat Official Account credentials and API IP whitelist;
* an OpenAI-compatible model endpoint for article generation;
* an OpenAlex API key if you plan to use the PAPER workflow.

The machine must be able to reach the QQ Bot gateway, configured RSS sources, OpenAlex, your
model endpoint, image sources, and the WeChat Official Account API for the integrations you use.

Clone and install
-----------------

.. code-block:: bash

   git clone https://github.com/thelightonmyway/wechat-news.git
   cd wechat-news
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   cp .env.example .env
   chmod 600 .env

Open ``.env`` in a local editor and replace the placeholders with credentials from your own
accounts. Never commit this file.

Process management
------------------

Run in the foreground:

.. code-block:: bash

   ./run.sh

Run in the background and manage the process:

.. code-block:: bash

   ./run.sh start
   ./run.sh status
   ./run.sh restart
   ./run.sh stop

``run.sh`` prefers ``.venv/bin/python``. In background mode it supervises the Bot process and
restarts it with backoff after an unexpected exit. Runtime logs and PID files are written under
``logs/`` and are excluded from Git.

Next steps
----------

Complete :doc:`qq-bot-setup`, :doc:`wechat-official-account-setup`, and
:doc:`model-configuration`, then follow :doc:`first-run`.
