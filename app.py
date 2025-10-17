import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import keras as kr

st.set_page_config(page_title="BrainScan", page_icon="🧠", layout="wide")

st.markdown("# Rapport de Projet - BrainScan")

st.markdown("""
1. Présentation
2. Chargement et Prétraitement du Dataset
3. Conception du Modéle CNN
4. Prédictions du Modéle
""")