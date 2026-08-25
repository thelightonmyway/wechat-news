Images and Copyright
====================

Separation of discovery and permission
--------------------------------------

Image discovery does not automatically make an image publishable. HTML extraction, newspaper4k,
Wikimedia Commons, NASA, OpenAlex metadata, and PDF extraction only produce candidate records.
``images/policy.py`` performs a separate conservative license and third-party-risk assessment.

NEWS image sources
------------------

The current NEWS generation workflow can create English search phrases from title-related text
and query:

* Wikimedia Commons;
* NOAA- and NASA-related results indexed through Wikimedia Commons;
* NASA Image and Video Library.

Candidates are filtered by metadata relevance, license, dimensions, aspect ratio, and non-content
markers. Logos, icons, advertising, tracking pixels, banners, and sprites are rejected during the
download stage. The current NEWS selector uses at most two non-redundant body images.

PAPER image sources
-------------------

The current PAPER workflow can use:

1. figures discovered in publisher HTML;
2. figures from an accessible open-access HTML location when the publisher page is unavailable;
3. numbered figures extracted from a formal or reference PDF when no legally usable HTML image
   is available;
4. the rendered first page of a downloaded paper PDF.

PDF figure extraction accepts a picture region only when it is large enough and adjacent to a
matching ``Fig. N`` caption. Supplementary and peer-review PDFs are excluded from PDF source
selection. The current PAPER selector uses at most four non-redundant body images.

Accepted license labels
-----------------------

The current policy accepts:

* CC BY
* CC BY-SA
* CC BY-NC
* CC BY-NC-SA
* CC0
* Public Domain

Rejected cases
--------------

The current policy rejects:

* NoDerivatives licenses, including CC BY-ND and CC BY-NC-ND;
* unknown or otherwise unconfirmed reusable licenses;
* ``all rights reserved``, publisher copyright, or copyrighted markers;
* captions or credits containing third-party risk markers such as reprinted, reproduced,
  with permission, third-party, or third party;
* Google Earth;
* Getty, Alamy, and Shutterstock.

Metadata and review
-------------------

The original caption, credit, license, provider, source URL, and policy reason are retained in the
article's internal ``metadata.json`` and SQLite image records. The body uses short generated image
captions rather than repeating long license or credit text.

Retaining internal metadata does not replace human review. Before final publication, verify that
the intended use is compatible with the image's actual license, required attribution, platform
rules, and applicable law.
