import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_ocr.postprocess.reranker import RERANKER

print('RERANKER OK', bool(RERANKER.model))
