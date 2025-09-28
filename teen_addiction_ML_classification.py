import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("Teen Smartphone Addiction  - ML Classifier Comparison")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("teen_phone_addiction_dataset.csv")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.write(df.head())

# Drop non-informative columns
non_informative_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'name']]
df = df.drop(columns=non_informative_cols, errors='ignore')

# Map Addiction_Level to categories
df["Addiction_Category"] = df["Addiction_Level"].apply(
    lambda x: "Not Addicted" if x <= 2 else ("Low" if x <= 4 else ("Moderate" if x <= 7 else "High"))
)

# Pie chart
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

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != "Addiction_Category":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Split features/target
X = df.drop(["Addiction_Level", "Addiction_Category"], axis=1)
y = LabelEncoder().fit_transform(df["Addiction_Category"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Supervised Models
supervised_models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="multinomial"),
    "kNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier()
}

# Train & evaluate supervised models
results = []
for name, model in supervised_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted")
    })

results_df = pd.DataFrame(results).set_index("Model")

# Confusion Matrix & Classification Report
st.subheader("Random Forest Evaluation")
col1, col2 = st.columns(2)

with col1:
    rf_model = supervised_models["Random Forest"]
    y_pred_rf = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_rf)
    fig_cm, ax_cm = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["High", "Moderate", "Low", "Not Addicted"],
                yticklabels=["High", "Moderate", "Low", "Not Addicted"],
                ax=ax_cm)
    ax_cm.set_ylabel("True Label")
    ax_cm.set_xlabel("Predicted Label")
    st.pyplot(fig_cm)

with col2:
    report_dict = classification_report(
        y_test, y_pred_rf, target_names=["High", "Moderate", "Low", "Not Addicted"], output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    classes_only = ["Not Addicted", "Low", "Moderate", "High"]
    report_df = report_df.loc[classes_only]
    st.dataframe(report_df)

# Feature Importance, Most & Least Important Feature 
if hasattr(rf_model, "feature_importances_"):
    st.subheader("Feature Importance Insights")
    feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_feat, ax_feat = plt.subplots(figsize=(4,4))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax_feat)
    ax_feat.set_xlabel("Importance")
    ax_feat.set_ylabel("")
    st.pyplot(fig_feat)
    
    col1, col2 = st.columns(2)
    with col1:
        most_imp_feat = feat_imp.index[0]
        st.subheader(f"Most Important: {most_imp_feat}")
        fig_most, ax_most = plt.subplots(figsize=(4,4))
        if df[most_imp_feat].nunique() <= 10:
            sns.countplot(x=most_imp_feat, hue="Addiction_Category", data=df, palette="Set2", ax=ax_most)
        else:
            sns.boxplot(x="Addiction_Category", y=most_imp_feat, data=df, palette="Set2", ax=ax_most)
        st.pyplot(fig_most)

    with col2:
        least_imp_feat = feat_imp.index[-1]
        st.subheader(f"Least Important: {least_imp_feat}")
        fig_least, ax_least = plt.subplots(figsize=(4,4))
        if df[least_imp_feat].nunique() <= 10:
            sns.countplot(x=least_imp_feat, hue="Addiction_Category", data=df, palette="Set2", ax=ax_least)
        else:
            sns.boxplot(x="Addiction_Category", y=least_imp_feat, data=df, palette="Set2", ax=ax_least)
        st.pyplot(fig_least)

# Unsupervised Clustering (K-Means)
st.subheader("Unsupervised Clustering (K-Means)")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df["Cluster"] = clusters
sil_score = silhouette_score(X, clusters)
st.write(f"Silhouette Score for K-Means: {sil_score:.3f}")

# Cluster distribution vs categories
fig9, ax9 = plt.subplots(figsize=(8,5))
sns.countplot(x="Cluster", hue="Addiction_Category", palette="Set3", data=df, ax=ax9)
ax9.set_title("Cluster Distribution vs Actual Addiction Categories")
st.pyplot(fig9)

# Model Performance Summary
st.subheader("Model Performance Summary (All Metrics)")
results_clean = results_df.round(3)
st.dataframe(results_clean.style.highlight_max(axis=0, color="lightgreen"))

st.subheader("Performance Metrics Comparison")
fig1, ax1 = plt.subplots(figsize=(12,6))
results_df.plot(kind="bar", ax=ax1)
plt.xticks(rotation=45)
plt.ylabel("Score")
plt.ylim(0, 1.05)
st.pyplot(fig1)

# Summary
best_model = results_df["Accuracy"].idxmax()
best_score = results_df["Accuracy"].max()

