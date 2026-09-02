"""Project-local configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_values(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def bind_qq_target_openid(openid: str, path: Path = ENV_FILE) -> tuple[bool, str]:
    """Atomically bind an empty QQ_TARGET_OPENID without overwriting a value."""
    target = openid.strip()
    if not target or "\n" in target or "\r" in target:
        raise ValueError("invalid QQ target OpenID")
    if not path.is_file():
        raise RuntimeError(f"configuration file does not exist: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "QQ_TARGET_OPENID":
            existing = value.strip().strip('"').strip("'")
            if existing:
                return False, existing

    updated: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key == "QQ_TARGET_OPENID":
                if not replaced:
                    updated.append(f"QQ_TARGET_OPENID={target}")
                    replaced = True
                continue
        updated.append(line)
    if not replaced:
        if updated and updated[-1]:
            updated.append("")
        updated.append(f"QQ_TARGET_OPENID={target}")

    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".env.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write("\n".join(updated).rstrip() + "\n")
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return True, target


@dataclass
class Settings:
    qq_app_id: str
    qq_client_secret: str
    qq_target_openid: str
    model_base_url: str
    model_api_key: str
    model_name: str
    openalex_api_key: str
    wiley_tdm_api_token: str
    wechat_app_id: str
    wechat_app_secret: str
    wechat_author: str
    daily_push_time: str
    daily_timezone: str
    database_path: Path

    @property
    def model_configured(self) -> bool:
        return bool(self.model_base_url and self.model_api_key and self.model_name)

    @property
    def openalex_configured(self) -> bool:
        return bool(self.openalex_api_key)

    @property
    def wechat_configured(self) -> bool:
        return bool(self.wechat_app_id and self.wechat_app_secret)


def load_settings() -> Settings:
    values = load_env_values()
    return Settings(
        qq_app_id=values.get("QQ_APP_ID", ""),
        qq_client_secret=values.get("QQ_CLIENT_SECRET", ""),
        qq_target_openid=values.get("QQ_TARGET_OPENID", ""),
        model_base_url=values.get("MODEL_BASE_URL", ""),
        model_api_key=values.get("MODEL_API_KEY", ""),
        model_name=values.get("MODEL_NAME", ""),
        openalex_api_key=values.get("OPENALEX_API_KEY", ""),
        wiley_tdm_api_token=values.get("WILEY_TDM_API_TOKEN", ""),
        wechat_app_id=values.get("WECHAT_APP_ID", ""),
        wechat_app_secret=values.get("WECHAT_APP_SECRET", ""),
        wechat_author=values.get("WECHAT_AUTHOR", ""),
        daily_push_time=values.get("DAILY_PUSH_TIME", "07:00"),
        daily_timezone=values.get("DAILY_TIMEZONE", "Asia/Shanghai"),
        database_path=PROJECT_ROOT / "data" / "news.db",
    )
