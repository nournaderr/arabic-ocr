from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_ocr.utils.image_io import load_image
from arabic_ocr.preprocess import preprocess
from arabic_ocr.segment import segment
from arabic_ocr.classifiers import get_classifier
from arabic_ocr.features.dot_features import dot_features
from arabic_ocr.postprocess import postprocess, ArabicLanguageModel
from arabic_ocr.postprocess.reranker import RERANKER

imgp = sys.argv[1] if len(sys.argv) > 1 else 'data/test_images/arabic2.jpg'
img = load_image(imgp)
bin = preprocess(img)
chars = segment(bin)
print('chars:', len(chars))

clf = get_classifier('cnn')
imgs = [c.img for c in chars]
dot_lists = [c.dots for c in chars]
all_cands = clf.predict_batch(imgs, dot_lists)

for crop, cands in zip(chars, all_cands):
    df = dot_features(crop.dots)
    crop.candidates = RERANKER.rerank(cands, df.tolist())

lm = ArabicLanguageModel()
text = postprocess(chars, lm)
print('final len', len(text))
print(text[:1000])
