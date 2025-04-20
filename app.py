import streamlit as st
import pickle
import matplotlib.pyplot as plt

#load model
with open('kmeans_model.pkl','rb') as f:
    model = pickle.load(f)
    
#set pafe cofig
st.set_page_config(page_title = 'k-Means Clustering App', layout = 'centered')
#set title
st.title("k-Means Clustering Visualizer")
#Diasplay
st.subheader('Example Data for Visualization')
st.markdown('This demo user example data (2D) to illustrate clustering results.')

#load form a saved dataset

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=model.n_clusters, cluster_std=0.60, random_state=0)
y_kmeans = model.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(loaded_model.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('k-Means Clustering')
plt.show()
