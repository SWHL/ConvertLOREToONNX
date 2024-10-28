# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2
import numpy as np

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

lineless_table_recognition = pipeline(
    Tasks.lineless_table_recognition,
    model="damo/cv_resnet-transformer_table-structure-recognition_lore",
)

img_path = "images/lineless_table_recognition.jpg"
result = lineless_table_recognition(img_path)

polygons = result["polygons"]
img = cv2.imread(img_path)
for poly in polygons:
    poly = np.array(poly).reshape(4, 2)
    cv2.polylines(img, np.int32([poly]), 1, (255, 0, 0))

cv2.imwrite("res.png", img)
print("ok")
