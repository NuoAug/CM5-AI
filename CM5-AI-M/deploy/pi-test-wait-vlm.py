"""Bounded local runner readiness check; no model input, external URL or redirect."""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def main():
    address = os.environ.get('VLM_URL', '')
    parsed = urllib.parse.urlsplit(address)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.port != 8091
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment
            or parsed.username or parsed.password):
        raise ValueError('Dedicated test requires VLM_URL=http://127.0.0.1:8091')
    key = os.environ.get('VLM_API_KEY', '')
    if not key or key in ('CHANGE_ME', 'REPLACE_WITH_RANDOM_LOCAL_KEY'):
        raise ValueError('Dedicated local VLM key is missing')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    def fetch_json(path):
        request = urllib.request.Request(address.rstrip('/') + path,
                                         headers={'Authorization': 'Bearer ' + key})
        with opener.open(request, timeout=2) as response:
            if response.status != 200:
                raise ValueError('Unexpected local model response')
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError('Local model metadata exceeds limit')
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError('Invalid local model metadata')
            return result
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            health = fetch_json('/health')
            if health.get('status') != 'ok':
                raise ValueError('Unexpected local model health response')
            data = fetch_json('/v1/models')
            if not any(isinstance(item, dict) and item.get('id') == 'minicpm-v'
                       for item in data.get('data', [])):
                raise ValueError('Local model identity is not minicpm-v')
            print('Dedicated MiniCPM model ready; physical audio/quality are not implied')
            return 0
        except urllib.error.HTTPError as error:
            if error.code != 503:
                raise ValueError('Local model identity/authentication check failed: HTTP ' + str(error.code)) from error
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise TimeoutError('Dedicated MiniCPM did not become ready within120s')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
