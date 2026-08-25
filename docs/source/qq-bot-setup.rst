QQ Bot Setup
============

Use your own QQ account and your own QQ Bot application. Never reuse another deployment's
AppID, secret, token, or OpenID.

Step 1 — Sign in to the QQ Bot platform
---------------------------------------

Open the official QQ Open Platform or QQ Bot management portal and sign in with the QQ account
that will own the Bot. Platform labels can change, but look for the product area used to create
and manage QQ Bots or applications.

Step 2 — Create a Bot
---------------------

Create a new Bot or application for your deployment. Complete any platform-required basic
information and enable the Bot for private user messages. wechat-news connects to the official
QQ Bot gateway and handles C2C private-message events.

Step 3 — Open the developer configuration
------------------------------------------

Open the Bot's developer, credentials, or application configuration page. Locate the values the
platform labels as AppID and AppSecret or ClientSecret.

Step 4 — Copy the AppID and secret locally
------------------------------------------

Add the credentials to the local ``.env`` file:

.. code-block:: ini

   QQ_APP_ID=YOUR_QQ_APP_ID
   QQ_CLIENT_SECRET=YOUR_QQ_APP_SECRET

The secret grants access to your Bot. Do not commit it, paste it into an Issue, include it in a
screenshot, or send it to another user. Reset it in the QQ platform immediately if it is exposed.

Step 5 — Configure the scheduled-delivery target
-------------------------------------------------

``QQ_TARGET_OPENID`` is the user OpenID that receives scheduled NEWS or PAPER candidate lists.
It is not your QQ number.

The current application supports automatic binding:

.. code-block:: ini

   QQ_TARGET_OPENID=

When this value is empty, the first user who sends the Bot a private message is atomically bound.
The OpenID is written into the local ``.env`` with restrictive file permissions. Once the value
is non-empty, the application does not overwrite it automatically.

For a single-user deployment, leaving it empty for the first controlled test is the simplest
setup. If several users can contact the Bot, set the intended OpenID before startup if you already
have it from your own QQ platform events or logs; the current Bot does not provide a command that
prints OpenIDs.

Step 6 — Start and verify
-------------------------

Start the service:

.. code-block:: bash

   ./run.sh start
   ./run.sh status

Open QQ, find your Bot, and send:

.. code-block:: text

   /ping

A working connection returns ``pong``. Then send ``/status`` to see the locally detected
configuration state. A configuration label confirms that values were loaded; it is not a full
external API test for model, OpenAlex, or WeChat access.
