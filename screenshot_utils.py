import os
import base64
import uuid
import re

SCREENSHOTS_FOLDER = os.path.join(os.getcwd(), 'screenshots')
os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)

def fix_base64_padding(b64_string):
    return b64_string + '=' * (-len(b64_string) % 4)

def save_base64_image(base64_str, output_dir=SCREENSHOTS_FOLDER):
    match = re.match(r'data:image/(?P<ext>[^;]+);base64,(?P<data>.+)', base64_str)
    if not match:
        raise ValueError("Invalid base64 image string")
    ext = match.group('ext')
    data = match.group('data')
    data += '=' * (-len(data) % 4)
    filename = f'screenshot_{uuid.uuid4().hex[:8]}.{ext}'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(data))
    return filename