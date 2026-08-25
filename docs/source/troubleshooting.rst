Troubleshooting
===============

The QQ Bot does not respond
---------------------------

* Run ``./run.sh status`` and confirm that both the supervisor and Bot process are running.
* Inspect ``logs/bot.log`` and ``logs/supervisor.log``.
* Confirm that ``QQ_APP_ID`` and ``QQ_CLIENT_SECRET`` belong to your own active Bot application.
* Confirm that the machine can reach the QQ Bot gateway.
* Send ``/ping``; a working Bot returns ``pong``.

The first user was bound incorrectly
------------------------------------

``QQ_TARGET_OPENID`` binds the first private-message sender when it is empty. Stop the Bot, edit
the local ``.env`` with the intended OpenID, and restart. The current application does not provide
a command that displays or changes OpenIDs.

Model generation is unavailable
-------------------------------

Check that ``MODEL_BASE_URL``, ``MODEL_API_KEY``, and ``MODEL_NAME`` are all set. ``/status`` reports
the model as configured only when all three are non-empty. If generation still fails, confirm that
the endpoint is reachable and supports the compatible chat-completions request used by the current
OpenAI Python client.

PAPER candidates are empty
--------------------------

* Confirm that ``OPENALEX_API_KEY`` is configured.
* OpenAlex timeouts and 429/5xx responses receive a limited number of retries; try again later if
  the service remains unavailable.
* A PAPER candidate must have a DOI, journal metadata, a valid publication date, a supported work
  type, and a match to the current scientific topic filters.

A publisher page returns 403
----------------------------

The PAPER workflow can try an open-access HTML location from OpenAlex metadata when the publisher
page is inaccessible. If no usable HTML image is available and a formal or reference PDF can be
found, generation can attempt PDF figure extraction. External site availability can still prevent
content or image retrieval.

No paper image is available
---------------------------

An image may be rejected because its license is unknown, contains a NoDerivatives or third-party
marker, is all-rights-reserved, cannot be downloaded, or does not meet content-image dimensions.
Inspect the generated article's ``metadata.json`` and the application logs. Do not bypass the
copyright policy merely to force an image into a draft.

WeChat status says configured but draft creation fails
------------------------------------------------------

``/status`` only checks whether ``WECHAT_APP_ID`` and ``WECHAT_APP_SECRET`` are non-empty. It does
not validate them against WeChat. Check:

* that the AppID and AppSecret belong to your own Official Account;
* that the account exposes the required media and draft API permissions;
* that the machine's public outbound IP is in the API IP whitelist;
* that the network can reach the WeChat API;
* that the selected article was generated before ``publish``.

WeChat error ``40164`` commonly indicates an IP whitelist mismatch.

The qihai header or cover is missing
------------------------------------

The included example header is ``assets/qihai-header.png`` and the default cover is
``assets/default-cover.jpg``. A PAPER first-page cover is available only when a source PDF was
successfully downloaded and rendered; otherwise the workflow falls back to an eligible selected
image or the default cover.

External APIs fail temporarily
-------------------------------

RSS publishers, image services, OpenAlex, the model endpoint, QQ, and WeChat are external systems.
Review the HTTP status and error type in local logs, confirm network and proxy settings, and retry
later. Never commit logs, downloaded files, debug output, or credentials to Git.
