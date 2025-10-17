import streamlit as st
from PIL import Image
import os

st.markdown("### Chargement et Prétraitement du Dataset")
st.markdown("#### 1. Importation et préparation des données")
st.markdown("""
- Chargement des images du dataset et vérification de leurs extensions (jpeg, jpg, bmp, png).
- Exploration des classes disponibles : les noms des dossiers représentent les catégories (glioma, meningioma, notumor, pituitary)
- Redimensionnement des images à une taille fixe (224x224) avec **OpenCV**.
""")

st.markdown("#### 2. Visualisation")

path = os.getcwd() + '/data/raw/Data'
classes = os.listdir(os.getcwd() + '/data/raw/Data')

for cls in classes:
    dir_path = os.path.join(path, cls)
    images = os.listdir(dir_path)

    st.text(cls)

    cols = st.columns(3)

    for col, img in zip(cols, images[:3]):
        img_resized = Image.open(dir_path + '/' + img)
        img_resized = img_resized.resize((200, 200))
        col.image(img_resized)

st.markdown("#### 3. Nombre d'images par classe.")

st.text('Counts')
st.table({
    'glioma': 1621,
    'meningioma': 1645,
    'notumor': 2000,
    'pituitary': 1757,
})

st.text('Percentage')
st.table({
    "glioma": "0.23%",
    "meningioma": "0.23%",
    "notumor": "0.28%",
    "pituitary": "0.25%",
})

st.image(os.getcwd() + '/output/graph1.png')

st.markdown("#### 4. Application du rééquilibrage")

st.markdown("""
on utilise la classe **ImageDataGenerator** de Keras pour effectuer une augmentation de données et équilibrer un jeu de données déséquilibré en générant de nouvelles images synthétiques pour les classes sous-représentées.
""")

st.markdown("##### 4.1 Configuration de l'augmentation")

st.markdown("""
Le générateur applique des transformations aléatoires légères sur les images existantes :
- rotations (±10°)
- décalages horizontaux et verticaux (10 %)
- zoom (±10 %)
- retournements horizontaux
- remplissage des zones vides par les pixels voisins
""")

st.markdown("##### 4.2 Équilibrage des classes")

st.markdown("""
- Le code identifie la classe possédant le plus grand nombre d'images.
- Pour chaque autre classe, il calcule le nombre d'images à générer pour atteindre la même taille d'échantillon.
""")

st.markdown("##### 4.3 Génération d'images")

st.markdown("""
- Les images d'une classe sont augmentées de manière itérative jusqu'à obtenir le nombre d'images souhaité.
- Les nouvelles images et leurs étiquettes sont ensuite concaténées au jeu de données initial (X, y).
""")


st.markdown("##### 4.4 Résultat final")

st.markdown("""
Le jeu de données résultant est équilibré entre toutes les classes et enrichi d'images artificielles, ce qui améliore la robustesse du modèle face à la variabilité visuelle.
""")

st.text("Counts:")
st.table({
    "glioma": 2000,
    "meningioma": 2000,
    "notumor": 2000,
    "pituitary": 2000,
})

st.text("Percentage:")
st.table({
    "glioma": '0.25%',
    "meningioma": '0.25%',
    "notumor": '0.25%',
    "pituitary": '0.25%',
})

st.image(os.getcwd() + '/output/graph2.png')


st.markdown("#### 5. Normalisation et encodage des labels")

st.markdown("#### 5.1. Normalisation des images")
st.text("Chaque image du jeu de données X est divisée par 255 pour mettre les valeurs des pixels dans la plage [0, 1], ce qui facilite l'entraînement du modèle.")


st.markdown("#### 5.2. Encodage des labels")
st.markdown("""
- Les étiquettes y sont d'abord transformées en entiers uniques à l'aide de **LabelEncoder**.
- Puis elles sont converties en vecteurs one-hot avec **to_categorical** pour être compatibles avec la sortie du modèle Keras en classification multi-classes.
""")

st.markdown("#### 6. Séparation du jeu de données")
st.markdown("""
Le code utilise train_test_split pour diviser le jeu de données en ensembles d'entraînement et de test :
- **80%%** des données sont utilisées pour l'entraînement (X_train, y_train)
- **20%%** pour les tests (X_test, y_test)
- L'option **stratify=y** garantit que la répartition des classes reste équilibrée dans les deux ensembles.
- **random_state=42** permet de reproduire la même division à chaque exécution.
""")




