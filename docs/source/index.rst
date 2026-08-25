.. meta::
   :description: Documentation for controlling a WeChat Official Account content workflow from QQ.

wechat-news
===========

**wechat-news** is a self-hosted automation tool for controlling a WeChat Official Account
content workflow from QQ. It runs on one continuously available computer or server and uses
a QQ Bot as the remote interface for discovery, selection, AI-assisted writing, image handling,
formatting, scheduling, and draft creation.

.. important::

   wechat-news creates drafts in your own WeChat Official Account. It does not automatically
   publish final posts.

The current default deployment focuses on scientific content. Some bundled sources, topic
filters, and the included ``qihai`` theme still reflect the author's deployment, while the
repository and account configuration are designed for other users to install with their own
QQ Bot, model service, and WeChat Official Account.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   introduction
   installation
   qq-bot-setup
   wechat-official-account-setup
   model-configuration
   configuration
   first-run

.. toctree::
   :maxdepth: 2
   :caption: Using wechat-news

   commands
   workflows
   images-and-copyright
   publishing
   scheduler
   updating

.. toctree::
   :maxdepth: 1
   :caption: Help

   troubleshooting
