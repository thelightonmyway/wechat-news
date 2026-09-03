Draft Creation
==============

The publishing command is deliberately a draft-creation command.

.. code-block:: text

   generate
   → Markdown article and metadata
   → qihai formatting
   → image upload
   → WeChat draft creation
   → manual review in the WeChat backend

Formatting
----------

wechat-news calls the vendored ``xiaohu-wechat-format`` formatter and uses the included ``qihai``
theme. If ``assets/qihai-header.png`` exists, the current deployment prepends that custom header
to the article during formatting.

The ``qihai`` theme and header are an included example from the author's deployment. This task
does not change their styling or remove them.

NEWS drafts
-----------

A NEWS draft uses the candidate's translated title when available, includes the generated article,
selected eligible images, and source information. The current image selector uses at most two body
images for NEWS content.

PAPER drafts
------------

A PAPER draft uses the complete Chinese paper title with a journal prefix and removes the first Markdown heading from
the formatted body, and can include up to four selected paper images. When a source PDF was
available and its first page was rendered, the paper first page is placed at the start of the
article and used as the WeChat cover. The body keeps an independent gray summary/lead box after the
first page, verified English excerpts, and normal Figure captions. Multi-panel PDF Figures are kept
as complete Figures rather than reduced to one panel.

Cover fallback
--------------

When a paper first-page cover is unavailable, the workflow uses a selected eligible cover image.
If no selected cover is available, it falls back to ``assets/default-cover.jpg``.

Dry-run behavior
----------------

If ``WECHAT_APP_ID`` and ``WECHAT_APP_SECRET`` are not both configured, ``publish`` completes the
local formatting step and returns the generated HTML path without creating a WeChat draft.

Final publication
-----------------

A successful command creates a draft through the WeChat Official Account API. It does not publish
the final post. Sign in to your own WeChat backend, review and edit the draft, and publish it
manually when ready.
