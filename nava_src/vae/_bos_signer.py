"""bos:// URL → pre-signed HTTPS URL.

Singleton BosClient so multi-worker data loaders don't pay re-handshake cost.
The signed URL is reusable; ffprobe / decord / torchaudio read https
directly, so we never touch the local disk.

Credentials and signing pattern mirror demo_fastapi (1).py:472-480 + 564-572
+ 636-638 verbatim — keep them in sync if those rotate.
"""
from threading import Lock

from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient

_BOS_HOST = "bj.bcebos.com"
_BOS_AK = ""
_BOS_SK = ""

_client = None
_lock = Lock()


def _get_client():
    """Lazily build the singleton BosClient. Double-checked locking is fine
    here because BosClient construction is independent and safe to repeat."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = BosClient(
                    BceClientConfiguration(
                        credentials=BceCredentials(_BOS_AK, _BOS_SK),
                        endpoint=_BOS_HOST,
                    )
                )
    return _client


def _split_bos_url(url: str):
    """``bos://bucket/dir/path/file.mp4`` → ``("bucket", "dir/path/file.mp4")``."""
    body = url[len("bos://"):]
    bucket, _, key = body.partition("/")
    return bucket, key


def resolve(url):
    """Pass-through unless ``url`` is a ``bos://`` URL.

    Sign bos:// URLs into HTTPS form so ffprobe / decord / torchaudio can read
    them directly. Local paths and HTTPS URLs are returned unchanged so callers
    can sprinkle this at every entry point cheaply.

    ``expiration_in_seconds=-1`` follows demo_fastapi's pattern (= max validity).
    """
    if not isinstance(url, str) or not url.startswith("bos://"):
        return url
    bucket, key = _split_bos_url(url)
    http = _get_client().generate_pre_signed_url(
        bucket, key, expiration_in_seconds=-1,
    )
    if isinstance(http, (bytes, bytearray)):
        http = http.decode("utf-8")
    return http
