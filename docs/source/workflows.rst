Workflows
=========

NEWS workflow
-------------

.. code-block:: text

   configured content sources
   → collection and topic filtering
   → deduplicated candidates
   → title translation and article generation
   → public-image discovery and image checks
   → WeChat formatting
   → draft creation

The current implementation collects configured RSS sources, normally using a 48-hour window and
expanding to 168 hours when too few topic-matched items are available. It deduplicates by DOI,
canonical URL, normalized title, and similar titles before ranking candidates.

When the model integration is available, it can translate candidate titles and assist selection.
Explicit ``generate`` commands use the model to write the selected article. NEWS image discovery
uses text metadata from public sources and applies the current license policy before download and
selection.

PAPER workflow
--------------

.. code-block:: text

   paper discovery
   → DOI and OpenAlex metadata verification
   → publisher or open-access article content
   → paper images, PDF figures, and first page when available
   → interpretation article generation
   → WeChat formatting
   → draft creation

The PAPER workflow requires an OpenAlex key in the current implementation. A candidate must have
a DOI and pass OpenAlex formal-publication checks, including publication date, journal metadata,
and a supported work type.

For content and images, the workflow tries the resolved publisher page first. When access fails,
it can try an open-access HTML location from existing metadata, preferring PubMed Central when
available. If no legally usable HTML image is available, generation can look for a formal or
reference PDF and use PyMuPDF4LLM to extract numbered figures. A downloaded PDF can also provide
the rendered first page and WeChat cover crop.

Manual control
--------------

Scheduled jobs deliver candidate lists. Users still choose a candidate and explicitly run
``generate`` and ``publish``. The scheduler does not generate articles, create drafts, or publish
final posts automatically.

Current limitations
-------------------

wechat-news is positioned as a reusable QQ-controlled WeChat workflow tool, but the present code
is not fully domain-neutral:

* the bundled RSS configuration focuses on science and climate-related sources;
* topic relevance and exclusion terms are currently hard-coded for the author's scientific use;
* generated writing is currently Chinese science communication and paper interpretation;
* the included ``qihai`` theme and header reflect the author's custom deployment.

These limitations are documented rather than hidden. Changing them requires a separate business-
logic task and is outside the public documentation rewrite.
