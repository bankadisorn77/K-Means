import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
#load model
with open('kmeans_model.pkl','rb') as f:
    model = pickle.load(f)
    

#set title
st.title("k-Means Clustering Visualizer by Adisorn Saard")

#set pafe cofig
st.set_page_config(page_title = 'k-Means Clustering App', layout = 'centered')

#Diasplay
st.subheader('Example Data for Visualization')
st.markdown('This demo user example data (2D) to illustrate clustering results.')
#cluster = st.slider('number of cluster',2,10)

#load form a saved dataset
X, _ = make_blobs(n_samples=300, centers=model.n_clusters, cluster_std=0.60, random_state=0)

#Prideict
y_kmeans = model.predict(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='red')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
