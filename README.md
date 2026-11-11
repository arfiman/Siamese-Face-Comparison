# What you need
1. Code Platform: Google Collaboration / Kaggle and VSCode
2. Programming Language: Python
3. Library: tensorflow, cv2, sklearn, os, numpy

# Dataset
No dataset is included in this repository. Collect your own face photos, with the consent of the people in them (or use a public face dataset within its licence terms). Do not scrape photos from the internet.

- **People:** photos of several different people. The notebooks split the dataset by person into training and test sets, so you need enough people for both.
- **Photos per person:** multiple. The model trains on triplets (anchor, positive, negative), so every person needs at least 2 photos. 10 or more is better, and the notebooks use up to 10 per person.
- **Variety:** different lighting, angles, expressions, accessories (glasses, hats) and days.
- **Quality:** one clearly visible face per photo.
- **Layout:** one folder per person, named with an ID rather than a real name, holding the cropped face images: `Extracted/<person_id>/<image>.jpg`
- **Cropping:** [2 Preprocess & Train Model.ipynb](https://github.com/arfiman/Siamese-Face-Comparison/blob/main/2%20Preprocess%20%26%20Train%20Model.ipynb) includes a helper (Haar cascade) that crops faces in place. It deletes the original files, so run it on a copy.
- **Path:** set `CUSTOM_ROOT` in the notebooks to your dataset folder.

# Development Step
![pipeline](https://github.com/verifikasiin/siamese-face-recognition-model/assets/65358140/d7a74c70-76bc-4424-8648-11aae2298d29)

# Steps for Training
1. Import data into the Code Platform
2. Resize the image into the desired size, we resize it to 224x224
3. Define the pre-trained model as the base model, we use a pre-trained model of VGG Face (Weights: https://www.kaggle.com/datasets/acharyarupak391/vggfaceweights)
![encoder model](https://github.com/verifikasiin/siamese-face-recognition-model/assets/65358140/7282f365-703d-4b51-bb18-c8a2a9fbf6bd)
  
4. Add flatten, Lambda layers as the top layer, Lambda used L2 Normalization method
5. Construct Siamese model with triplet loss to train encoder model
6. Define loss function and optimizer.
7. Train the model
8. After obtaining the desired accuracy, we can convert the model into a keras saved model
9. (Optional) Compress the model to tflite using tf.float16 option for quantization
```
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
```
For the complete code checkout the notebook on https://github.com/verifikasiin/siamese-face-recognition-model/blob/main/2%20Preprocess%20%26%20Train%20Model.ipynb

# Steps for Predicting
![inference](https://github.com/verifikasiin/siamese-face-recognition-model/assets/65358140/b6c14b5e-f4d4-4e1f-ade0-5d3658f04abf)

1. Load the tflite model using read byte and allocate tensor for the interpreter
  ```
   with open(model_path, 'rb') as f:
    model_tflite = f.read()
  interpreter = tf.lite.Interpreter(model_content=model_tflite)
  interpreter.allocate_tensors()
   ```
2. Read image then detect and crop face from image
3. Resize the face image to 224x224
4. Invoke the interpreter for the encoder model
5. Measure the distance between encoded value of two image

For the complete code on inference, check out the python script https://github.com/verifikasiin/siamese-face-recognition-model/blob/main/5%20face_verification_lite.py 
