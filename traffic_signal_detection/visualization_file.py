import pickle
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from xgboost import XGBClassifier

with open("../signal_detection/archive/test.p", "rb") as file:
    data = pickle.load(file)

images = data["features"]
names = data["labels"]

#RGB --> Greyscale -> extracting features
grey_scale_img = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in images]
hog_features = [hog(i, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for i in grey_scale_img]



model = XGBClassifier()
model.load_model("xgb_model.json")

predictions = model.predict(hog_features)

def visualize_predictions(images, actual_labels, predicted_labels, num_samples=5) -> None:
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Actual: {actual_labels[i]}"
                  f"\nPred: {predicted_labels[i]}")

        plt.axis("off")
    plt.show()

visualize_predictions(grey_scale_img, names, predictions)
