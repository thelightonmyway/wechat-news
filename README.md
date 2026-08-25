# wechat-news

> Control your WeChat Official Account workflow from QQ.

**wechat-news** runs on one continuously available computer or server. QQ acts as the remote controller: use a QQ Bot to retrieve content candidates, generate or translate articles with an AI model, handle images, format posts, and create drafts in your own WeChat Official Account.

The workflow is designed for review rather than unattended final publishing. It creates WeChat drafts; you inspect, edit, and publish them manually from the WeChat backend.

```text
QQ Bot
→ content discovery and selection
→ AI-assisted translation and writing
→ image handling
→ WeChat formatting
→ scheduling
→ WeChat draft creation
```

The included default sources and topic filters currently focus on scientific content. The bundled `qihai` theme and `assets/qihai-header.png` are an example custom deployment used by the author, not the identity or required branding of the project.

## Features

- QQ-based remote control
- Separate NEWS and PAPER workflows
- Configured RSS and content collection
- AI-assisted title translation and article writing through an OpenAI-compatible API
- DOI and OpenAlex metadata for published papers
- Publisher HTML and open-access article extraction
- Article and paper image handling
- Conservative image license and copyright filtering
- PDF figure extraction with PyMuPDF4LLM fallback
- Paper first-page rendering
- Included `qihai` WeChat formatting theme
- Scheduled candidate delivery
- WeChat Official Account draft creation

## QQ Commands

| Command | Description |
|---|---|
| `/ping` | Check whether the bot is online |
| `/news` | Retrieve or display today's NEWS candidates |
| `/news N` | Show details for NEWS candidate `N` |
| `/news N generate` | Generate an article for NEWS candidate `N` |
| `/news N publish` | Format the generated NEWS article and create a WeChat draft |
| `/papers` | Retrieve or display today's PAPER candidates |
| `/paper N` | Show paper metadata and image information for candidate `N` |
| `/paper N generate` | Generate an interpretation article for PAPER candidate `N` |
| `/paper N publish` | Format the generated PAPER article and create a WeChat draft |
| `/status` | Show candidate, integration, delivery, and recent-error status |
| `/history` | Show recent generated, drafted, and failed records |

## Quick Start

Python 3.10 or newer is required.

```bash
git clone https://github.com/thelightonmyway/wechat-news.git
cd wechat-news
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
# Edit .env and add your own QQ, model, OpenAlex, and WeChat credentials.
./run.sh start
./run.sh status
```

For the complete account setup and first-run guide, see the documentation.

## Documentation

Full documentation: <https://wechat-news.readthedocs.io/en/latest/>

## Open Source Reuse

See [OPEN_SOURCE_REUSE.md](OPEN_SOURCE_REUSE.md) for the main reused libraries, services, licenses, and vendored component attribution.

## Acknowledgements

WeChat formatting and draft integration reuse the vendored [xiaohu-wechat-format](vendor/xiaohu-wechat-format/) project. The repository also includes the author's custom `qihai` theme as an example deployment theme.

## Version

v0.1.0
