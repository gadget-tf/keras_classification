# coding: utf-8

import coremltools

model_path = './model/model.h5'
labels_path = './model/labels.txt'

coreml_model = coremltools.converters.keras.convert(
    model_path,
    input_names='image',
    image_input_names='image',
    class_labels=labels_path,
    is_bgr=True,
    image_scale=1./255
)

coreml_model.save('./model.mlmodel')