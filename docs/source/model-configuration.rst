Model Configuration
===================

Supported interface
-------------------

wechat-news uses the OpenAI Python client against an OpenAI-compatible API endpoint. You may use
a model service of your choice as long as it provides the compatible chat-completions interface
expected by the current client code.

The project is not tied to one model vendor.

Required values
---------------

Configure all three values in the local ``.env``:

.. code-block:: ini

   MODEL_BASE_URL=https://api.example.com/v1
   MODEL_API_KEY=YOUR_MODEL_API_KEY
   MODEL_NAME=YOUR_MODEL_NAME

``MODEL_BASE_URL``
   The base URL published by your chosen compatible service. Use a URL reachable from the machine
   running wechat-news.

``MODEL_API_KEY``
   Your key for that service. Treat it as a secret and never commit or publish it.

``MODEL_NAME``
   The model identifier accepted by the service.

All three values must be non-empty. If any one is missing, the project reports the model as not
configured and refuses explicit article generation.

What the model is used for
--------------------------

The current code can use the configured model for:

* NEWS candidate title translation and candidate selection;
* image-search keyword generation;
* short Chinese image captions from text metadata;
* AI-assisted NEWS and PAPER article writing.

The application sends text and metadata to the model. Its writing prompt produces Chinese
science communication or paper interpretation content in the current deployment. This public
documentation change does not alter that behavior.

Safe configuration
------------------

Do not copy a private proxy URL, loopback endpoint from another machine, or another user's API
key into public files. ``https://api.example.com/v1`` is documentation-only. Replace it locally
with your own provider's endpoint.

Verification
------------

Send ``/status`` to confirm that all three values were loaded. Then use a real candidate:

.. code-block:: text

   /news N generate

or:

.. code-block:: text

   /paper N generate

Generation success verifies the endpoint, key, model name, network path, and API compatibility.
No separate model test command exists in the current application.
