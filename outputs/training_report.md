# Rapport d'Entrainement — RavenEncoderDecoder

## 1. Objectif

Entrainer le modele encodeur-decodeur `RavenEncoderDecoder` sur **toutes les 7 configurations** du dataset RAVEN pour apprendre a extraire les proprietes structurelles (presence, type, taille, couleur, angle, position) des images RPM (Raven's Progressive Matrices).

Le modele servira ensuite d'extracteur de features pour le GNN de raisonnement.

---

## 2. Configuration de l'entrainement

| Parametre | Valeur |
|-----------|--------|
| **Modele** | `RavenEncoderDecoder` (2,465,411 parametres) |
| **GPU** | NVIDIA GeForce RTX 5060 Laptop |
| **Epoques** | 20 |
| **Batch size** | 64 |
| **Learning rate** | 1e-4 |
| **Optimiseur** | Adam |
| **Perte** | CompositeLoss (BCE + CrossEntropy + MSE + Reconstruction) |
| **Poids reconstruction** | 20.0 |
| **Duree totale** | ~22 min |

### Donnees

| Split | Panels | Fichiers NPZ | Limite par config |
|-------|--------|---------------|-------------------|
| **Train** | 22,400 | 1,400 | 200/config |
| **Validation** | 22,400 | 1,400 | 200/config |
| **Test** | 22,400 | 1,400 | 200/config |

### Les 7 configurations utilisees

1. `center_single` — 1 objet au centre
2. `distribute_four` — 4 objets distribues
3. `distribute_nine` — 9 objets distribues
4. `in_center_single_out_center_single` — 2 composants In/Out
5. `in_distribute_four_out_center_single` — composant In distribue + Out centre
6. `left_center_single_right_center_single` — composants gauche/droite
7. `up_center_single_down_center_single` — composants haut/bas

---

## 3. Courbes d'apprentissage

![Courbes d'apprentissage du RavenEncoderDecoder](C:/Users/dhias/.gemini/antigravity/brain/06e650b6-33e4-4bce-a8ac-83bc4f04ff12/learning_curves.png)

### Analyse des courbes

- **Perte totale** : Convergence nette de 3.51 a 0.42 (train) et 2.66 a 0.69 (val). Chute marquee autour de l'epoque 10 indiquant que le modele a capture la structure principale.
- **Perte presence (BCE)** : Decroit rapidement de 0.17 a 0.01, montrant que le modele detecte tres bien la presence/absence d'objets.
- **Perte attributs (CE + MSE)** : Reduction progressive de 1.41 a 0.08. Les types et attributs continus sont bien appris.
- **Perte reconstruction (MSE)** : Plateau a 0.075 pendant les 9 premieres epoques (poids du decoder pas encore significatifs), puis chute rapide a 0.016. Le decoder reconstruit les images de maniere correcte.

> [!NOTE]
> L'ecart train/val reste modere (~0.27 a l'epoque 20), ce qui indique une generalisation raisonnable sans overfitting severe.

---

## 4. Resultats sur le jeu de test

### Pertes

| Composante | Valeur |
|------------|--------|
| **Perte totale** | 0.6904 |
| Perte presence | 0.0277 |
| Perte attributs | 0.3150 |
| Perte reconstruction | 0.0174 |

### Metriques de performance

| Metrique | Score |
|----------|-------|
| **Precision presence** | **98.88%** |
| **Precision type** | **87.91%** |
| MAE taille | 0.1257 |
| MAE couleur | 0.1204 |
| MAE angle | 0.2650 |
| MSE reconstruction | 0.0174 |

### Interpretation

- **Presence (98.88%)** : Le modele detecte quasi-parfaitement si un slot contient un objet. Cela signifie que le backbone CNN extrait efficacement les features de presence.

- **Type (87.91%)** : Bonne precision pour la classification des types de formes (triangle, carre, pentagone, hexagone, cercle). L'erreur de ~12% pourrait etre amelioree avec plus d'epoques ou un dataset plus grand.

- **Taille/Couleur (MAE ~0.12)** : Valeurs normalisees entre 0 et 1, donc une MAE de 0.12 represente environ 12% d'erreur relative. C'est un bon resultat pour un premier entrainement.

- **Angle (MAE 0.265)** : L'angle est l'attribut le plus difficile a predire, ce qui est attendu car les rotations subtiles sont visuellement moins distinctes que les changements de type ou couleur.

- **Reconstruction (MSE 0.017)** : Le decoder reconstruit les images avec une erreur tres faible, confirmant que l'espace latent (10 slots x 13 features) capture les informations essentielles.

---

## 5. Architecture du modele

```
RavenEncoderDecoder (2,465,411 params)
├── backbone (CNN)
│   ├── Conv2d(1, 32, 3) + BN + ReLU
│   ├── Conv2d(32, 64, 3) + BN + ReLU + MaxPool
│   ├── Conv2d(64, 128, 3) + BN + ReLU + MaxPool
│   └── AdaptiveAvgPool2d → (128,)
├── projection_layer (Linear 128 → 10*13 = 130)
│   → reshape → z: (B, 10, 13) — vecteur structurel
└── decoder (Transposed CNN)
    ├── Linear(130, 128*5*5)
    ├── ConvTranspose2d(128, 64) + BN + ReLU
    ├── ConvTranspose2d(64, 32) + BN + ReLU
    ├── ConvTranspose2d(32, 16) + BN + ReLU
    └── ConvTranspose2d(16, 1) + Sigmoid
        → x_hat: (B, 1, 160, 160) — reconstruction
```

### Vecteur structurel z (10 slots x 13 features)

| Colonne | Contenu | Type |
|---------|---------|------|
| 0 | Presence (0/1) | Binaire |
| 1 | Number (layout) | Regression |
| 2-7 | Type (one-hot sur 6 classes) | Classification |
| 8 | Size | Regression [0, 1] |
| 9 | Color | Regression [0, 1] |
| 10-11 | Position X, Y (bbox) | Regression |
| 12 | Angle | Regression [0, 1] |

---

## 6. Fichiers generes

| Fichier | Description |
|---------|-------------|
| [learning_curves.png](file:///c:/Users/dhias/workspaces/these/HRI-GNN/outputs/learning_curves.png) | Courbes d'apprentissage (4 sous-graphiques) |
| [training_history.json](file:///c:/Users/dhias/workspaces/these/HRI-GNN/outputs/training_history.json) | Historique detaille des pertes par epoque |
| [test_metrics.json](file:///c:/Users/dhias/workspaces/these/HRI-GNN/outputs/test_metrics.json) | Metriques d'evaluation sur le jeu de test |
| [raven_encoder_decoder_final.pth](file:///c:/Users/dhias/workspaces/these/HRI-GNN/outputs/raven_encoder_decoder_final.pth) | Poids du modele complet |
| [raven_encoder_final.pth](file:///c:/Users/dhias/workspaces/these/HRI-GNN/outputs/raven_encoder_final.pth) | Poids de l'encodeur seul (pour le GNN) |
| Checkpoints (epoques 5, 10, 15, 20) | Sauvegardes intermediaires |

---

## 7. Pistes d'amelioration

1. **Plus de donnees** : Entrainer sur l'integralite du dataset (78K fichiers au lieu de 1.4K) pourrait ameliorer significativement les resultats.
2. **Plus d'epoques** : La perte diminuait encore a l'epoque 20, il y a de la marge pour continuer.
3. **Scheduler de learning rate** : Ajouter un `CosineAnnealingLR` ou `ReduceLROnPlateau` pour affiner la convergence.
4. **Data augmentation** : Rotations/flips pour ameliorer la prediction d'angle.
5. **Attention spatiale** : Remplacer l'AdaptiveAvgPool par un mecanisme d'attention pour mieux localiser les objets multiples.
