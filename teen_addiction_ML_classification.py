import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("Teen Smartphone Addiction")


@st.cache_data
def load_data():
    df = pd.read_csv("teen_phone_addiction_binary.csv")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.write(df.head())


non_informative_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'name']]
df = df.drop(columns=non_informative_cols, errors='ignore')

df["Addiction_Category"] = df["Addiction_Level"].apply(
    lambda x: "Not Addicted" if x <= 2 else ("Low" if x <= 4 else ("Moderate" if x <= 7 else "High"))
)

st.subheader("Addiction Category Distribution")
fig0, ax0 = plt.subplots()
df["Addiction_Category"].value_counts().plot.pie(
    autopct="%1.1f%%", startangle=90, ax=ax0, colors=sns.color_palette("Set2")
)
ax0.set_ylabel("")
st.pyplot(fig0)

# Gender vs Daily Usage
fig6, ax6 = plt.subplots(figsize=(8,5))
sns.barplot(data=df, x='Gender', y='Daily_Usage_Hours', ci='sd', palette='pastel', ax=ax6)
ax6.set_title('Average Daily Phone Usage Hours by Gender')
ax6.set_ylabel('Daily Usage Hours')
ax6.set_xlabel('Gender')
st.pyplot(fig6)

# Daily Screen Time & Time on Social Media
st.subheader("Screen Time Analysis")
col1, col2 = st.columns(2)

with col1:
    fig_screen, ax_screen = plt.subplots(figsize=(5,4))
    sns.histplot(df['Daily_Usage_Hours'], kde=True, bins=20, color='skyblue', ax=ax_screen)
    ax_screen.set_title('Distribution of Daily Screen Time')
    ax_screen.set_xlabel('Daily Usage Hours')
    ax_screen.set_ylabel('Frequency')
    ax_screen.grid(True)
    st.pyplot(fig_screen)

with col2:
    fig_social, ax_social = plt.subplots(figsize=(5,4))
    sns.histplot(df['Time_on_Social_Media'], kde=True, bins=20, color='lightgreen', ax=ax_social)
    ax_social.set_title('Distribution of Time on Social Media')
    ax_social.set_xlabel('Time on Social Media')
    ax_social.set_ylabel('Frequency')
    ax_social.grid(True)
    st.pyplot(fig_social)

# Daily Usage vs Addiction Level
grouped_df = df.groupby('Daily_Usage_Hours')['Addiction_Level'].mean().reset_index()
fig8, ax8 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=grouped_df, x='Daily_Usage_Hours', y='Addiction_Level', marker='o', color='blue', ax=ax8)
ax8.set_title('Daily Usage Hours vs Average Addiction Level')
ax8.set_xlabel('Daily Usage Hours')
ax8.set_ylabel('Average Addiction Level')
ax8.grid(True)
fig8.tight_layout()
st.pyplot(fig8)

# Encode categorical features except Addiction_Category
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != "Addiction_Category":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le


X_multi = df.drop(["Addiction_Level", "Addiction_Level_Category", "Addiction_Category"], axis=1)
y_multi = LabelEncoder().fit_transform(df["Addiction_Category"])

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

scaler_multi = StandardScaler()
X_train = scaler_multi.fit_transform(X_train)
X_test = scaler_multi.transform(X_test)


supervised_models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="multinomial"),
    "kNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier()
}


results = []
for name, model in supervised_models.items():
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Overfitting / Underfitting check
    if train_acc - test_acc > 0.05:
        fit_status = "Overfitting (High Variance)"
    elif test_acc - train_acc > 0.05:
        fit_status = "Underfitting (High Bias)"
    else:
        fit_status = "Good Fit"
    
    results.append({
        "Model": name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Precision": precision_score(y_test, y_test_pred, average="weighted"),
        "Recall": recall_score(y_test, y_test_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_test_pred, average="weighted"),
        "Fit Status": fit_status
    })

results_df = pd.DataFrame(results).set_index("Model")

st.subheader("Model Performance")
st.dataframe(results_df.style.highlight_max(subset=["Test Accuracy"], axis=0, color="lightgreen"))

fig_fit, ax_fit = plt.subplots(figsize=(10,6))
results_df[["Train Accuracy","Test Accuracy"]].plot(kind="bar", ax=ax_fit)
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy (Bias/Variance Analysis)")
st.pyplot(fig_fit)

# Random Forest Feature Importance
st.subheader("Random Forest Feature Importance")
rf_model = supervised_models["Random Forest"]
if hasattr(rf_model, "feature_importances_"):
    feat_imp = pd.Series(rf_model.feature_importances_, index=X_multi.columns).sort_values(ascending=False)
    fig_feat, ax_feat = plt.subplots(figsize=(4,4))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax_feat)
    ax_feat.set_xlabel("Importance")
    ax_feat.set_ylabel("")
    st.pyplot(fig_feat)


st.subheader("Boxplots for Most & Least Important Features")
most_imp_feature = feat_imp.index[0]
least_imp_feature = feat_imp.index[-1]

col1, col2 = st.columns(2)

with col1:
    fig_box1, ax_box1 = plt.subplots(figsize=(5,4))
    sns.boxplot(data=df, y=most_imp_feature, x='Addiction_Category', palette="Set2", ax=ax_box1)
    ax_box1.set_title(f'Boxplot of Most Important Feature: {most_imp_feature}')
    st.pyplot(fig_box1)

with col2:
    fig_box2, ax_box2 = plt.subplots(figsize=(5,4))
    sns.boxplot(data=df, y=least_imp_feature, x='Addiction_Category', palette="Set2", ax=ax_box2)
    ax_box2.set_title(f'Boxplot of Least Important Feature: {least_imp_feature}')
    st.pyplot(fig_box2)


st.subheader("Binary Classification Metrics for Random Forest")

y_binary = df["Addicted"].apply(lambda x: 1 if x in [1, "Yes", "True"] else 0)
X_bin = df.drop(["Addiction_Level", "Addiction_Level_Category", "Addiction_Category", "Addicted"], axis=1)

for col in X_bin.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_bin[col] = le.fit_transform(X_bin[col].astype(str))

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

scaler_bin = StandardScaler()
X_train_bin = scaler_bin.fit_transform(X_train_bin)
X_test_bin = scaler_bin.transform(X_test_bin)

rf_bin = RandomForestClassifier()
rf_bin.fit(X_train_bin, y_train_bin)
y_pred_bin = rf_bin.predict(X_test_bin)
y_prob_bin = rf_bin.predict_proba(X_test_bin)[:,1]

cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
tn, fp, fn, tp = cm_bin.ravel()

st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(
    cm_bin,
    index=["Actual Negative", "Actual Positive"],
    columns=["Predicted Negative", "Predicted Positive"]
)
st.dataframe(cm_df.style.background_gradient(cmap="Blues"))

# Metrics
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
f1 = 2 * (precision * tpr) / (precision + tpr)
roc_auc = roc_auc_score(y_test_bin, y_prob_bin)

metrics_df = pd.DataFrame({
    "Metric": ["TPR (Recall/Sensitivity)", "TNR (Specificity)", "FPR", "FNR", "Accuracy", "Precision", "F1 Score", "ROC-AUC Score"],
    "Value": [tpr, tnr, fpr, fnr, accuracy, precision, f1, roc_auc]
}).round(3)
st.subheader("Binary Classification Metrics")
st.dataframe(metrics_df)

# ROC-AUC and ROC Curve
roc_auc = roc_auc_score(y_test_bin, y_prob_bin)
fpr_curve, tpr_curve, thresholds = roc_curve(y_test_bin, y_prob_bin)

fig_roc, ax_roc = plt.subplots(figsize=(6,5))
ax_roc.plot(fpr_curve, tpr_curve, color='blue', label=f'AUC = {roc_auc:.3f}')
ax_roc.plot([0,1], [0,1], color='red', linestyle='--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve (Addicted)')
ax_roc.legend(loc='lower right')
st.pyplot(fig_roc)


