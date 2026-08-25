WeChat Official Account Setup
==============================

Use your own WeChat Official Account. Do not use credentials, screenshots, or account details
from the author's deployment.

Your account must expose the API capabilities needed to obtain an access token, upload media,
and manage drafts. Availability can depend on account type, verification status, and current
WeChat platform policy. Check the API permissions shown in your own account backend rather than
assuming every account has the same access.

Step 1 — Prepare your WeChat Official Account
---------------------------------------------

Sign in to the official WeChat Official Account management platform at
`mp.weixin.qq.com <https://mp.weixin.qq.com/>`_ with the administrator account for the Official
Account that should receive drafts.

Confirm that this is your own target account and that its API permissions include the draft and
media operations required by the workflow.

Step 2 — Open developer or API settings
---------------------------------------

Open the developer settings area. Depending on the current interface, this is commonly named
**Developer**, **Basic Configuration**, **Development Configuration**, or a similar API settings
page. Use the function names rather than relying on an old screenshot or exact menu position.

Step 3 — Obtain the AppID
-------------------------

Copy the AppID displayed for your Official Account and place it in the local ``.env``:

.. code-block:: ini

   WECHAT_APP_ID=YOUR_WECHAT_APP_ID

Step 4 — Obtain or reset the AppSecret
--------------------------------------

Use the same developer configuration area to obtain or reset the AppSecret. Store it only in the
local ``.env``:

.. code-block:: ini

   WECHAT_APP_SECRET=YOUR_WECHAT_APP_SECRET

Treat the AppSecret as a password:

* never upload it to GitHub;
* never paste it into a GitHub Issue;
* never include it in screenshots or public logs;
* reset it immediately in the WeChat platform if it is exposed.

Step 5 — Configure the API IP whitelist
---------------------------------------

The WeChat API sees the public outbound IP address of the machine running wechat-news. Add that
address to the Official Account API IP whitelist in the developer configuration area.

* Local computer deployment: use the public outbound IP of the network that computer uses.
* Cloud server deployment: use the server's public outbound IP.

Do not enter a private LAN address such as ``192.168.x.x`` or a loopback address. If the public
outbound IP changes, update the whitelist before using the API again.

An incorrect or missing whitelist commonly causes access-token requests to fail or returns an
IP-not-allowed / whitelist-related API error. WeChat error ``40164`` is a common example.

Step 6 — Set the author
-----------------------

Set the author name that should be passed to draft creation:

.. code-block:: ini

   WECHAT_AUTHOR=YOUR_AUTHOR_NAME

The included formatting setup also uses this value as its author label. It is not an API secret.

Step 7 — Verify local configuration
-----------------------------------

Start the Bot and send:

.. code-block:: text

   /status

The WeChat line reports configured when both ``WECHAT_APP_ID`` and ``WECHAT_APP_SECRET`` are
non-empty. This verifies local configuration loading only. The first real draft operation verifies
that the AppID, AppSecret, account API permissions, network access, and IP whitelist work together.

Step 8 — Create and inspect a draft
-----------------------------------

Generate an article first, then create a draft:

.. code-block:: text

   /news N generate
   /news N publish

or:

.. code-block:: text

   /paper N generate
   /paper N publish

A successful ``publish`` command creates a draft in your own WeChat Official Account. It does
not publish the final post. Sign in to the WeChat backend, open **Drafts**, inspect and edit the
article, and publish it manually only when it is ready.

If the WeChat credentials are empty, the same command performs a local formatting dry-run and
does not call the draft API.
