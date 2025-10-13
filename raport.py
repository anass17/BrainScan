import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import keras as kr

st.title("les bibliothèques nécessaires:")
st.markdown("""
- **Pandas** : Librairie pour la manipulation et l'analyse de données sous forme de tableaux.
- **NumPy** : Outil pour le calcul numérique rapide avec des tableaux et des matrices.
- **OpenCV** : Librairie pour le traitement et l'analyse d'images et de vidéos.
- **Matplotlib** : Outil de visualisation pour créer des graphiques et des courbes.
- **Streamlit** : Framework pour créer des applications web interactives en Python.
- **TensorFlow** : Framework de deep learning pour entraîner et déployer des modèles d'IA.
- **Keras** : Interface simple pour concevoir et entraîner des réseaux de neurones avec TensorFlow.
- **os** : Module intégré pour interagir avec le système de fichiers et les répertoires.
""")