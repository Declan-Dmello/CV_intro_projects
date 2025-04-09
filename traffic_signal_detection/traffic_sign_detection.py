import pickle
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


with open("../signal_detection/archive/train.p", "rb") as file:
    data = pickle.load(file)

print(type(data))
print(data.keys())


images = data["features"]
coords = data["coords"]
names = data["labels"]
sizes  =  data["sizes"]


print(images.shape)

#converting to greyscale
grey_scale_img = [cv2.cvtColor(i ,cv2.COLOR_RGB2GRAY) for i in images]

#extracting features
hog_features = [hog(i, pixels_per_cell=(8,8), cells_per_block=(2,2)) for i in grey_scale_img]

X = hog_features
y =  names

X_train , X_test , y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2 , random_state=42)
print("Reached here")
model = XGBClassifier(tree_method="hist", device="cuda")

model.fit(X_train,y_train,eval_set=[(X_test, y_test)], eval_metric="mlogloss")

predictions = model.predict(X_test)

print(accuracy_score(y_test , predictions))
model.save_model("xgb_model.json")



