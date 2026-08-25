First Run
=========

This checklist takes a new deployment from account preparation to a draft visible in the user's
own WeChat Official Account.

1. Create your QQ Bot
---------------------

Sign in to the official QQ Bot platform, create a Bot, and obtain its AppID and AppSecret or
ClientSecret. See :doc:`qq-bot-setup`.

2. Prepare your WeChat Official Account
---------------------------------------

Sign in to your own account at ``mp.weixin.qq.com``. Obtain the AppID and AppSecret, confirm the
required API permissions, and add the deployment machine's public outbound IP to the API IP
whitelist. See :doc:`wechat-official-account-setup`.

3. Prepare a compatible model API
---------------------------------

Choose an OpenAI-compatible model service and obtain its base URL, API key, and model name. See
:doc:`model-configuration`.

4. Prepare OpenAlex for PAPER
-----------------------------

Obtain an OpenAlex API key if you want to use the PAPER workflow. NEWS-only use can leave this
integration empty, but the current PAPER candidate discovery and formal-publication verification
require it.

5. Clone the repository
-----------------------

.. code-block:: bash

   git clone https://github.com/thelightonmyway/wechat-news.git
   cd wechat-news

6. Create the Python environment
--------------------------------

.. code-block:: bash

   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt

7. Create the local configuration
---------------------------------

.. code-block:: bash

   cp .env.example .env
   chmod 600 .env

Edit ``.env`` and replace the placeholders with credentials from your own accounts. Clear any
optional integration that you do not intend to use. Never copy another deployment's credentials.

8. Start the service
--------------------

.. code-block:: bash

   ./run.sh start
   ./run.sh status

``status`` should show the supervisor and Bot process running. If startup fails, inspect
``logs/bot.log`` and ``logs/supervisor.log``.

9. Verify the QQ connection
---------------------------

Open QQ, find your Bot, and send:

.. code-block:: text

   /ping

The Bot should return ``pong``. If ``QQ_TARGET_OPENID`` was empty, this first private message also
binds the sender as the scheduled-delivery target.

10. Check integration status
----------------------------

Send:

.. code-block:: text

   /status

Confirm that the model, OpenAlex, and WeChat lines match the integrations you configured. These
labels confirm that values were loaded; actual generation and draft creation test the external
services.

11. Retrieve candidates
-----------------------

For NEWS:

.. code-block:: text

   /news

For PAPER:

.. code-block:: text

   /papers

The first request can take longer because it may collect sources, extract content, filter topics,
query metadata services, and populate the local database.

12. Inspect a candidate
-----------------------

.. code-block:: text

   /news N

or:

.. code-block:: text

   /paper N

Replace ``N`` with a rank from the candidate list.

13. Generate the article
------------------------

.. code-block:: text

   /news N generate

or:

.. code-block:: text

   /paper N generate

The Bot returns the local Markdown path after successful generation.

14. Create the WeChat draft
---------------------------

.. code-block:: text

   /news N publish

or:

.. code-block:: text

   /paper N publish

If the WeChat integration is configured correctly, the Bot reports that a draft was created. If
credentials are absent, it reports a local formatting dry-run instead.

15. Review and publish manually
-------------------------------

Sign in to your own WeChat Official Account backend, open **Drafts**, find the generated article,
and inspect the title, author, cover, body, images, and source information. Edit anything that
needs correction. Publish it manually only when you are satisfied.

wechat-news does not automatically perform the final publication step.
