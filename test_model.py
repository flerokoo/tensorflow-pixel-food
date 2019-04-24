from prepare_data import get_image_data
from tensorflow.keras import models
import numpy as np



(X, Y) = get_image_data("test_images")

model = models.load_model("trained.model")

predictions = model.predict([X])
predictions = list(map(lambda x: np.argmax(x), predictions))

right_answers = 0
for (i, prediction) in enumerate(predictions):
    if Y[i] == prediction:
        right_answers += 1

print(f"Correct answer percentage: {right_answers/len(Y) * 100}%")    
    


