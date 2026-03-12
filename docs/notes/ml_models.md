
# Classic ML

| Modèle                | Réseau de neurones | Type                | Spécificité                 | Biais inductif                     | Usages                         |
| --------------------- | ------------------ | ------------------- | --------------------------- | ---------------------------------- | ------------------------------ |
| Régression linéaire   | ❌ Non              | ML classique        | relation linéaire           | relation linéaire entre variables  | économie, prévision simple     |
| Régression logistique | ❌ Non              | ML classique        | classification probabiliste | séparation linéaire                | scoring crédit, fraude         |
| Arbres de décision    | ❌ Non              | ML classique        | règles hiérarchiques        | partition de l’espace des features | classification interprétable   |
| Random Forest         | ❌ Non              | Ensemble learning   | ensemble d’arbres           | réduction de variance              | tabular data                   |
| Gradient Boosting     | ❌ Non              | Ensemble learning   | arbres séquentiels          | correction progressive des erreurs | finance, marketing             |
| SVM                   | ❌ Non              | ML classique        | hyperplan optimal           | marge maximale                     | classification haute dimension |
| KNN                   | ❌ Non              | ML classique        | voisinage                   | similarité locale                  | recommandation simple          |
| Naive Bayes           | ❌ Non              | probabiliste        | modèle bayésien             | indépendance des variables         | spam detection                 |
| K-Means               | ❌ Non              | clustering          | partition en clusters       | clusters sphériques                | segmentation                   |
| PCA                   | ❌ Non              | réduction dimension | projection linéaire         | maximisation de variance           | compression données            |
## Probabilistic Models

|Modèle|Réseau de neurones|Spécificité|Biais inductif|Usages|
|---|---|---|---|---|
|Hidden Markov Model|❌ Non|états cachés|dépendance Markov|speech recognition|
|Bayesian Networks|❌ Non|graph probabiliste|causalité|diagnostic|
|Gaussian Processes|❌ Non|distribution sur fonctions|smoothness|regression avancée|
# Deep Learning - ANN (Artificial Neural Network)

| Modèle                     | Réseau de neurones | Type          | Spécificité                           | Biais inductif                   | Usages              |
| -------------------------- | ------------------ | ------------- | ------------------------------------- | -------------------------------- | ------------------- |
| Feedforward Neural Network | ✅ Oui              | Deep Learning | couches entièrement connectées        | combinaison additive de features | tabular data        |
| Autoencoder                | ✅ Oui              | DL            | compression / reconstruction          | représentation latente compacte  | détection anomalies |
| Variational Autoencoder    | ✅ Oui              | DL génératif  | génération probabiliste               | espace latent continu            | génération données  |
| GAN                        | ✅ Oui              | DL génératif  | compétition générateur/discriminateur | réalisme des données générées    | images synthétiques |
| Diffusion Models           | ✅ Oui              | DL génératif  | génération par débruitage progressif  | processus de diffusion           | génération d’images |

# Deep Learning - ANN for images

| Modèle                            | Réseau de neurones | Spécificité                  | Biais inductif                  | Usages                |
| --------------------------------- | ------------------ | ---------------------------- | ------------------------------- | --------------------- |
| **Convolutional Neural Networks** | ✅ Oui              | convolution spatiale         | équivariance aux translations   | vision par ordinateur |
| ResNet                            | ✅ Oui              | connexions résiduelles       | apprentissage profond stable    | vision avancée        |
| U-Net                             | ✅ Oui              | architecture encoder-decoder | structure spatiale hiérarchique | segmentation médicale |
| Vision Transformer                | ✅ Oui              | attention globale            | relations longues distances     | vision moderne        |

# Deep Learning - ANN for sequences

|Modèle|Réseau de neurones|Spécificité|Biais inductif|Usages|
|---|---|---|---|---|
|Recurrent Neural Networks|✅ Oui|mémoire séquentielle|dépendance temporelle|texte, séries temporelles|
|LSTM|✅ Oui|mémoire longue|dépendances longues|NLP, finance|
|GRU|✅ Oui|RNN simplifié|mémoire efficace|NLP|
|Temporal CNN|✅ Oui|convolution temporelle|motifs locaux temporels|forecasting|
# Deep Learning - Transformers

|Modèle|Réseau de neurones|Spécificité|Biais inductif|Usages|
|---|---|---|---|---|
|Transformer|✅ Oui|self-attention|interactions globales|NLP|
|BERT|✅ Oui|transformer encoder|compréhension bidirectionnelle|NLP|
|GPT-4|✅ Oui|transformer génératif|prédiction autoregressive|LLM|
|T5|✅ Oui|texte → texte|multitâche universel|NLP|

# Deep Learning - Graph Machine Learning

| Modèle                    | Réseau de neurones | Spécificité            | Biais inductif                | Usages                  |
| ------------------------- | ------------------ | ---------------------- | ----------------------------- | ----------------------- |
| **Graph Neural Networks** | ✅ Oui              | message passing        | équivariance aux permutations | réseaux sociaux         |
| GCN                       | ✅ Oui              | convolution sur graphe | voisinage local               | classification de nœuds |
| GraphSAGE                 | ✅ Oui              | agrégation inductive   | généralisation nouveaux nœuds | recommandation          |
| GAT                       | ✅ Oui              | attention sur graphe   | importance des voisins        | graph learning          |
