from PIL import Image
import PIL.ExifTags as ExifTags


class ExifImage:

    def __init__(self, fname):
        # 画像ファイルを開く --- (*1)
        self.img = Image.open(fname)
        self.exif = {}
        if self.img._getexif():
            for k, v in self.img._getexif().items():
                if k in ExifTags.TAGS:
                    self.exif[ExifTags.TAGS[k]] = v
 
    def print(self):
        if self.exif:
            for k, v in self.exif.items():
                print(k, ":", v)
        else:
            print("exif情報は記録されていません。")
a = ExifImage("data/train/0a0da090aa9f0342444a7df4dc250c66.jpg")
a.print()