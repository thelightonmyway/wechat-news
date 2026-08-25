Introduction
============

What the project does
---------------------

wechat-news turns one continuously running computer or server into a remotely controlled
WeChat Official Account content workstation. You send commands to your own QQ Bot; the local
service performs the requested work and returns candidate lists, status information, and
results through QQ.

.. code-block:: text

   QQ Bot
   → content discovery and selection
   → AI-assisted translation and writing
   → image handling
   → WeChat formatting
   → scheduling
   → WeChat draft creation

The project intentionally stops at draft creation. You remain responsible for reviewing,
editing, and publishing the final post in the WeChat Official Account backend.

Who can deploy it
-----------------

Any user who can provide the following can run their own deployment:

* a QQ Bot application and its credentials;
* a WeChat Official Account with the API permissions needed for access tokens, media uploads,
  and draft management;
* an OpenAI-compatible text model endpoint for generation features;
* an OpenAlex API key when using the PAPER workflow;
* a Linux, WSL, or server environment with Python 3.10 or newer.

Content modes
-------------

NEWS
~~~~

The NEWS workflow collects entries from configured RSS and content sources, applies the current
topic filters, separates candidates, generates or translates an article, handles eligible
images, formats the result, and creates a WeChat draft.

PAPER
~~~~~

The PAPER workflow validates formally published papers with DOI and OpenAlex metadata, extracts
article or paper content, handles publisher HTML figures and PDF fallbacks, generates an
interpretation article, formats it, and creates a WeChat draft.

Current scope and example deployment
------------------------------------

The current source list and topic relevance rules are oriented toward scientific and climate-
related content because they reflect the author's active deployment. They are real application
constraints, not a claim that the software is already domain-neutral. This documentation does
not change those rules.

The bundled ``qihai`` formatter theme and ``assets/qihai-header.png`` are an included custom
example used by the author's deployment. They are not the public identity of wechat-news and
are not credentials or user-specific account configuration.
