Updating
========

wechat-news does not currently include a dedicated update command or update script. Use the
standard Git workflow from the project directory.

Before updating
---------------

Check whether tracked files have local modifications:

.. code-block:: bash

   cd ~/wechat-news
   git status --short

Commit, stash, or otherwise preserve intentional tracked changes before pulling. Do not use
``git reset --hard`` as a routine update method.

Pull and refresh dependencies
-----------------------------

.. code-block:: bash

   git pull --ff-only origin main
   .venv/bin/python -m pip install -r requirements.txt

``--ff-only`` prevents Git from creating an unexpected merge commit. The local ``.env`` is
ignored by Git and remains in place, but you should compare ``.env.example`` after updates in case
new configuration variables were added.

Restart and verify
------------------

.. code-block:: bash

   ./run.sh restart
   ./run.sh status

Then send ``/ping`` and ``/status`` through QQ.

Updating a fresh clone
----------------------

If you intentionally prefer a clean deployment, clone into a new directory, create a new virtual
environment, and copy configuration values manually from your private local ``.env``. Never copy
credentials into Git-tracked files.
