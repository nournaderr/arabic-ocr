from arabic_ocr.pipeline import ArabicOCRPipeline
import numpy as np

pipe = ArabicOCRPipeline.__new__(ArabicOCRPipeline)

class Broken:
    def predict_batch(self, imgs, dots=None):
        raise RuntimeError('boom')

pipe.classifier = Broken()
pipe.lm = None
pipe.debug = False

img = np.full((60,200,3),255,dtype=np.uint8)
img[10:50,20:180] = 0

print('Running pipeline with broken classifier...')
print(pipe.run_array(img))
