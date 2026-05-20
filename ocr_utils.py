import os
import shutil

import pytesseract


DEFAULT_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.join(
        os.path.expanduser("~"),
        "AppData",
        "Local",
        "Programs",
        "Tesseract-OCR",
        "tesseract.exe",
    ),
]


class OCRConfigurationError(RuntimeError):
    """Raised when Tesseract OCR is not installed or cannot be located."""


def configure_tesseract():
    """Locate Tesseract and configure pytesseract to use it."""
    candidates = []

    env_path = os.environ.get("TESSERACT_PATH")
    if env_path:
        candidates.append(env_path)

    path_lookup = shutil.which("tesseract")
    if path_lookup:
        candidates.append(path_lookup)

    candidates.extend(DEFAULT_TESSERACT_CANDIDATES)

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            candidate = os.path.join(candidate, "tesseract.exe")

        if candidate and os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    raise OCRConfigurationError(
        "Tesseract OCR is not installed or could not be found. "
        "Install Tesseract and either add it to PATH or set the TESSERACT_PATH "
        "environment variable to the full tesseract.exe path."
    )
