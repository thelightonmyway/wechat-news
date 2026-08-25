# Open Source Reuse

wechat-news keeps its current application structure and the existing `vendor/xiaohu-wechat-format` integration. This document lists the main directly used libraries, services, and vendored components. Exact terms are governed by the license and package metadata of the installed version.

| Purpose | Project or service | Integration | License or usage note |
|---|---|---|---|
| RSS parsing | feedparser | Python dependency | BSD-2-Clause |
| Article extraction | trafilatura | Python dependency | Apache-2.0 |
| News metadata and image discovery | newspaper4k | Python dependency | MIT |
| HTML parsing | Beautiful Soup | Python dependency | MIT |
| HTTP clients | aiohttp / HTTPX | Python dependencies | Apache-2.0 / BSD-3-Clause |
| Paper metadata | PyAlex / OpenAlex | Python dependency and OpenAlex API | PyAlex is MIT; OpenAlex data is subject to its service terms |
| Scheduling | APScheduler | Python dependency | MIT |
| PDF rendering and extraction | PyMuPDF / PyMuPDF4LLM | Python dependencies | AGPL-3.0 or commercial licensing; review the applicable terms before redistribution or hosted deployment |
| Model API client | OpenAI Python client | OpenAI-compatible API client | Apache-2.0 |
| Markdown processing | Python-Markdown | Python dependency | BSD-3-Clause |
| WeChat formatting and drafts | xiaohu-wechat-format | Vendored under `vendor/xiaohu-wechat-format` | The upstream README declares MIT; upstream README files and attribution are retained |
| Public image discovery | Wikimedia Commons | Per-image metadata and license checks | Each image has its own license and attribution requirements |
| Public image discovery | NASA Image and Video Library | API and media metadata | Media use depends on NASA guidance and each item's copyright metadata; content is not automatically treated as public domain |

## Vendored formatting component

`vendor/xiaohu-wechat-format` remains ordinary vendored source rather than a Git submodule. The project includes local formatting adaptations and the `qihai` custom theme used by the author's deployment.

The runtime-generated `vendor/xiaohu-wechat-format/config.json` is excluded from Git because it can contain WeChat credentials. Real AppIDs, AppSecrets, API keys, tokens, and OpenIDs must remain in local ignored configuration files.
