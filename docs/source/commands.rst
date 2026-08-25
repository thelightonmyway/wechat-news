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
     - Format the already generated PAPER article and create a WeChat draft.
   * - ``/status``
     - Show candidate counts, recent collection state, scheduled delivery state, integration
       configuration, and the most recent error.
   * - ``/history``
     - Show the 10 most recent generated, drafted, and failed records.

Command rules
-------------

* ``N`` is the rank shown in the current candidate list.
* ``generate`` requires ``MODEL_BASE_URL``, ``MODEL_API_KEY``, and ``MODEL_NAME``.
* ``publish`` never runs ``generate`` implicitly. Generate the selected article first.
* If WeChat credentials are missing, ``publish`` performs a formatting dry-run and returns the
  generated HTML path.
* A successful ``publish`` command creates a draft only. It does not publish the final post.
* The current code does not implement a ``next`` command.

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
   /history
