from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_ocr.pipeline import ArabicOCRPipeline


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/arabic2.jpg"
    pipe = ArabicOCRPipeline(classifier="cnn", debug=True)
    try:
        txt = pipe.run(img)
        print('RESULT LENGTH', len(txt))
        print(txt[:1000])
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
