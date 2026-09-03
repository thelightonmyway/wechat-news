QQ Commands
===========

Send commands as private messages to your QQ Bot.

Command overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``/ping``
     - Check whether the Bot is online; returns ``pong``.
   * - ``/news``
     - Retrieve or display today's NEWS candidates.
   * - ``/news N``
     - Extract and display details for NEWS candidate ``N``.
   * - ``/news N generate``
     - Generate an article for NEWS candidate ``N`` with the configured model.
   * - ``/news N publish``
     - Format the already generated NEWS article and create a WeChat draft.
   * - ``/papers``
     - Retrieve or display today's PAPER candidates.
   * - ``/paper N``
     - Display paper, DOI, OpenAlex, and image information for candidate ``N``.
   * - ``/paper N generate``
     - Generate an interpretation article for PAPER candidate ``N``.
   * - ``/paper N publish``
     - Generate PAPER candidate ``N`` and then create a WeChat draft.
   * - ``/paperurl <URL或DOI>``
     - Identify a paper, generate its PAPER article, and create a WeChat draft.
   * - ``/status``
     - Show candidate counts, recent collection state, scheduled delivery state, integration
       configuration, and the most recent error.
   * - ``/history``
     - Show the 10 most recent generated, drafted, and failed records.

Command rules
-------------

* ``N`` is the rank shown in the current candidate list.
* ``generate`` requires ``MODEL_BASE_URL``, ``MODEL_API_KEY``, and ``MODEL_NAME``.
* For PAPER, ``/paper N publish`` runs ``generate`` first and creates the draft only after generation
  succeeds. ``/paper N generate`` remains available as a standalone generation/debug entry point.
* ``/paperurl <URL或DOI>`` performs the same generate-then-draft flow for a direct paper request.
* NEWS ``publish`` still expects the selected NEWS article to have been generated first.
* If WeChat credentials are missing, ``publish`` performs a formatting dry-run and returns the
  generated HTML path.
* A successful ``publish`` command creates a draft only. It does not publish the final post.
* ``/papers next`` requests another PAPER candidate batch while preserving the current list if the
  refresh fails.

Examples
--------

.. code-block:: text

   /ping
   /status
   /news
   /news 2
   /news 2 generate
   /news 2 publish

   /papers
   /paper 1
   /paper 1 generate
   /paper 1 publish
   /paperurl 10.1029/2025GL120559
   /history
