import pandas as pd

# load trail datasets
df1 = pd.read_csv("Trail1_extracted_features_acceleration_m1ai1-1.csv")
df2 = pd.read_csv("Trail2_extracted_features_acceleration_m1ai1.csv")
df3 = pd.read_csv("Trail3_extracted_features_acceleration_m2ai0.csv")

# combine into one dataframe
df = pd.concat([df1, df2, df3], ignore_index=True)

# quick checks
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Event labels:", df["event"].unique())
print(df.head())


# remove non-feature columns
df = df.drop(columns=["start_time", "axle", "cluster", "tsne_1", "tsne_2"])

# convert event labels to binary
df["event"] = df["event"].apply(lambda x: 0 if x == "normal" else 1)

# check result
print("\nAfter cleaning:")
print("Shape:", df.shape)
print("Event value counts:\n", df["event"].value_counts())


# separate features and target
X = df.drop("event", axis=1)
y = df["event"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


from sklearn.preprocessing import StandardScaler

# normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nAfter normalization:")
print("First row (scaled):\n", X_scaled[0])


from sklearn.model_selection import train_test_split

# split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# train SVM model
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train, y_train)

# predictions
y_pred = svm_model.predict(X_test)

# evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nSVM Accuracy:", accuracy)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(svm_model, X_scaled, y, cv=5)

print("\nCross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())


from sklearn.feature_selection import SelectKBest, f_classif

# select top 5 features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y)

# get selected feature names
selected_features = X.columns[selector.get_support()]

print("\nSelected features (ANOVA):", list(selected_features))


# use only selected features
X_selected = df[selected_features]

# normalize selected features
X_selected_scaled = scaler.fit_transform(X_selected)

# train-test split again
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_selected_scaled, y, test_size=0.2, random_state=42
)

# train SVM on selected features
svm_selected = SVC(kernel="rbf", random_state=42)
svm_selected.fit(X_train_s, y_train_s)

# evaluate
y_pred_s = svm_selected.predict(X_test_s)
accuracy_selected = accuracy_score(y_test_s, y_pred_s)

print("\nSVM Accuracy with selected features:", accuracy_selected)


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                              display_labels=["Normal", "Defect"])
disp.plot()
plt.title("Confusion Matrix – SVM Track Defect Detection")
plt.tight_layout()
plt.show()