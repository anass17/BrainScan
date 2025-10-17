import streamlit as st
import pandas as pd
import os

st.markdown("### Conception du Modele CNN")
st.markdown("#### 1. Définition du modèle CNN")

st.markdown("""
Le code définit un réseau de neurones convolutif (CNN) pour la classification d'images à l'aide de Keras :

1. Couches convolutives et de pooling :

- Trois blocs **Conv2D** → **MaxPooling2D** → **SpatialDropout2D** pour extraire progressivement les caractéristiques visuelles.
- Les filtres augmentent à chaque bloc (32 → 64 → 128) pour capturer des détails de plus en plus complexes.
- **SpatialDropout2D(0.25)** réduit le surapprentissage en désactivant aléatoirement certaines cartes de caractéristiques.

2. Flatten et couches denses :

- La couche Flatten transforme les cartes de caractéristiques 2D en vecteur 1D.
- Une couche Dense de 128 neurones avec sigmoid suit, avec un **Dropout(0.5)** pour réduire le surapprentissage.

3. Couche de sortie :

- La couche finale **Dense(4, activation='softmax')** produit les probabilités pour les 4 classes de sortie.

4. Résumé général :
            
Ce CNN est conçu pour extraire des caractéristiques hiérarchiques des images et effectuer une classification multi-classes avec régularisation pour améliorer la généralisation.
""")

st.markdown("#### 2. Entraînement du modèle CNN")

st.markdown("""
1. Compilation du modèle :

- **Optimiseur** : Adam avec un learning rate de 0.001
- **Fonction de perte** : categorical_crossentropy (pour classification multi-classes)
- **Métrique** : accuracy

2. Callbacks pour un entraînement efficace :

- **EarlyStopping** : arrête l'entraînement si la perte de validation ne s'améliore pas pendant 5 époques et restaure les meilleurs poids.
- **ModelCheckpoint** : enregistre le meilleur modèle (selon la perte de validation) dans le répertoire ../models/.

3. Paramètres d'entraînement :

- **Taille de batch** : 32
- **Nombre d'époques prévu** : 40
- **Validation interne** : 10%% du jeu d'entraînement

4. Durée et résultat :

- Le modèle a atteint sa meilleure performance après 14 époques.
- **Temps total d'entraînement** : 14 minutes 28 secondes.

5. Résumé général :
            
L'entraînement utilise des techniques de régularisation et de sauvegarde automatique pour améliorer la généralisation et éviter le surapprentissage.
""")


st.text("Training History:")

history = {
    "Epoch": list(range(1, 15)),
    "Time_s": [62, 59, 83, 58, 60, 58, 68, 60, 61, 59, 59, 59, 59, 59],
    "Accuracy": [0.5903, 0.7510, 0.7997, 0.8406, 0.8681, 0.8828, 0.9102, 0.9170, 0.9352, 0.9448, 0.9536, 0.9613, 0.9646, 0.9715],
    "Loss": [0.9471, 0.6097, 0.5008, 0.4203, 0.3528, 0.3073, 0.2492, 0.2177, 0.1804, 0.1541, 0.1334, 0.1155, 0.0976, 0.0851],
    "Val_Accuracy": [0.7422, 0.7906, 0.8000, 0.8625, 0.8687, 0.8500, 0.8813, 0.8906, 0.8906, 0.8734, 0.8703, 0.8922, 0.9047, 0.9109],
    "Val_Loss": [0.6045, 0.4877, 0.4687, 0.4025, 0.3482, 0.3600, 0.3293, 0.2956, 0.2760, 0.3397, 0.3641, 0.3220, 0.2875, 0.2878]
}

df = pd.DataFrame(history)

st.dataframe(df, hide_index=True)


st.text("Accurracy et loss de testing: ")

st.table({
    "Loss" : 0.2727,
    "Accuracy" : 0.9050,
})

st.text("Visualiser les courbes d'apprentissage:")

st.image(os.getcwd() + '/output/graph3.png')


st.text("Matrice de confusion:")

st.image(os.getcwd() + '/output/graph4.png')