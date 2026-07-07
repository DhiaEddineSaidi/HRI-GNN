
**Induction Hiérarchique de Règles (Hierarchical Rule Induction - HRI)** C'est le cœur de la contribution scientifique. Le modèle HRI améliore les limites des approches précédentes **LRI (Logical Rule Induction)** grâce à plusieurs innovations :

- **Proto-règles :** Ce sont des versions étendues et adaptatives des méta-règles classiques. Elles permettent de représenter un grand nombre de règles potentielles de manière très compacte, indépendamment du nombre exact d'arguments (l'arité) des prédicats.
- **Structure hiérarchique :** Les prédicats sont organisés en couches successives, allant des faits initiaux (couche 0) jusqu'au résultat final. Une règle d'une couche donnée ne peut être construite qu'à partir des prédicats des couches inférieures, ce qui impose un biais d'apprentissage fort et réduit considérablement les coûts de calcul.
- **Inférence améliorée :** Les auteurs modifient les opérations mathématiques (utilisation de fonctions POOL, MERGE, etc.) pour éviter un problème de "sous-estimation" des probabilités qui limitait le modèle LRI précédent.
- Ils prouvent mathématiquement que leur petit ensemble de proto-règles génériques (appelé R0​ ou R∗) est **extrêmement expressif** et suffisant pour générer une vaste catégorie de clauses logique

# ILP (Inductive Logic Programming)

| Terme                                             | Définition selon la Section 3.1                                                                      | Exemple tiré du document                                                                                                                                 |
| :------------------------------------------------ | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Constantes (Constants)**                        | Les objets du discours de l'univers logique défini.                                                  | `0`, `1`, `2` dans le domaine des nombres.                                                                                                               |
| **Fonctions (Functions)**                         | Applications mathématiques transformant des constantes en une nouvelle constante.                    | _Note : Le document précise qu'il se concentre sur un fragment logique "sans fonction" (function-free) pour simplifier le problème._                     |
| **Prédicats (Predicates)**                        | Applications prenant des constantes et renvoyant une valeur booléenne (Vrai ou Faux).                | `Zero`, `Succ` (successeur), `Even` (pair) ou `Aux` (auxiliaire).                                                                                        |
| **Atome (Atom)**                                  | Un prédicat appliqué à des arguments (constantes ou variables).                                      | `Succ(X, Z)` ou `Even(X)` où X et Z sont des variables non spécifiées.                                                                                   |
| **Atome ancré (Ground atom)**                     | Un atome dont tous les arguments sont exclusivement des constantes.                                  | `Zero(0)` ou `Succ(0, 1)`.                                                                                                                               |
| **Littéral (Literal)**                            | **Un atome ou la négation d'un atome**.                                                              | `Even(X)` (littéral positif) ou `¬Even(X)` (littéral négatif).                                                                                           |
| **Formules FOL (FOL formulas)**                   | Combinaisons d'atomes avec des connecteurs logiques et des quantificateurs.                          | L'utilisation de symboles comme $\exists$ (il existe), $\forall$ (pour tout), $\land$ (ET logique) ou $\leftarrow$ (implication) pour relier des atomes. |
| **Clauses**                                       | Formules logiques écrites sous la forme d'une disjonction ("OU" logique) de **littéraux**.           | $\neg P \lor \neg Q \lor U$ (Non P, OU Non Q, OU U).                                                                                                     |
| **Clauses de Horn (Horn clauses)**                | Clauses contenant au maximum un seul littéral positif.                                               | $False \leftarrow P \land Q$ (zéro littéral positif) ou $U \leftarrow P \land Q$ (un littéral positif).                                                  |
| **Clauses définies (Definite clauses)**           | Clauses de Horn avec exactement un littéral positif, pouvant s'écrire comme une règle d'implication. | La règle $Even(X) \leftarrow Zero(X)$.                                                                                                                   |
| **Atome de tête (Head atom)**                     | La conclusion $H$ dans une règle d'implication ($H \leftarrow B_1 \wedge \dots \wedge B_k$).         | `Even(X)` dans la règle $Even(X) \leftarrow Zero(X)$.                                                                                                    |
| **Atome de corps (Body atom)**                    | Les conditions $B_i$ dans une règle d'implication.                                                   | `Zero(X)` dans la règle $Even(X) \leftarrow Zero(X)$.                                                                                                    |
| **ILP (et son objectif)**                         | Trouver un ensemble de règles déduisant tous les exemples positifs et aucun exemple négatif.         | Trouver les règles définissant les nombres pairs à partir d'exemples positifs (0, 2, 4...) et négatifs (1, 3, 5...).                                     |
| **Faits (Facts)**                                 | Atomes ancrés (vérités factuelles) contenus dans les connaissances de base.                          | `Zero(0)`, `Succ(0, 1)`, et `Succ(1, 2)`.                                                                                                                |
| **Prédicats d'entrée (Input predicates)**         | Prédicats apparaissant directement dans les faits des connaissances de base.                         | `Zero` et `Succ`.                                                                                                                                        |
| **Prédicats auxiliaires (Auxiliary predicates)**  | Nouveaux prédicats inventés pour aider à définir le concept cible.                                   | `Aux`, inventé pour faire le lien entre deux sauts de nombres : $Aux(X,Y) \leftarrow Succ(X,Z) \land Succ(Z,Y)$.                                         |
| **Prédicat extensionnel (Extensional predicate)** | Prédicat défini concrètement par un ensemble d'atomes ancrés.                                        | `Succ`, car il est défini par une liste de faits : `{Succ(0, 1), Succ(1, 2)...}`.                                                                        |
| **Prédicat intensionnel (Intensional predicate)** | Prédicat défini de manière abstraite par des règles logiques (clauses).                              | `Even`, car il est défini par des règles comme $Even(X) \leftarrow Zero(X)$.                                                                             |

# Predicate Invention and Meta-Rules 

**1. L'invention de prédicats (Predicate Invention)** Pour résoudre un problème complexe, il est souvent nécessaire de créer des étapes intermédiaires. L'invention de prédicats consiste pour le modèle à créer automatiquement des **prédicats auxiliaires** (définis de manière abstraite par des règles) qui l'aideront finalement à définir le prédicat cible qu'il cherche à apprendre.

**2. Le besoin d'un "Biais de Langage" et les Méta-règles** Si l'on laissait la machine chercher toutes les combinaisons possibles de règles, l'espace de recherche grandirait de façon exponentielle et le calcul serait impossible. Pour éviter cela, on utilise des **méta-règles** (ou modèles de règles) qui imposent des contraintes de syntaxe.

- _Exemple :_ Une méta-règle de "chaînage" s'écrit $H(X,Y) \leftarrow B_1(X,Z) \land B_2(Z, Y)$. Ici, $H$, $B_1$ et $B_2$ ne sont pas des prédicats spécifiques, mais des **variables de prédicats** (des "trous" à remplir). La machine sait qu'elle doit chercher une règle qui a exactement cette forme.
# Logical Rule Induction LRI 

Voici un récapitulatif étape par étape du fonctionnement mathématique du modèle LRI, en intégrant les symboles et les équations officiels du document :

**Étape 1 : Représentation vectorielle (Embeddings)** La machine ne manipule plus des mots ou des symboles stricts, mais des listes de nombres (des vecteurs).

- Chaque prédicat $P$ (comme _Succ_ ou _Zero_) est représenté par un vecteur de dimension $d$, noté **$\theta_P \in \mathbb{R}^d$**.
- Dans une méta-règle (par exemple la règle de chaînage $H(X,Y) \leftarrow B_1(X,Z) \land B_2(Z, Y)$), les variables qui servent de "trous" ($H, B_1, B_2$) reçoivent elles aussi des vecteurs : **$\theta_H, \theta_{B1}, \theta_{B2}$**.
- Un fait concret, ou atome ancré $P(s, o)$, est stocké sous la forme **$(\theta_P, s, o, v_P)$**, où $s$ et $o$ sont les arguments (sujet, objet), et **$v_P \in [0,1]$** est la _valuation_, c'est-à-dire le degré de certitude que ce fait est vrai.

**Étape 2 : Unification Souple (Soft Unification)** Pour savoir quel prédicat insérer dans quel "trou" de la règle, la machine calcule la similarité (la ressemblance) entre leurs vecteurs.

- **$\alpha_{PH} = \cos(\theta_P, \theta_H)$**.
- Ce score $\alpha_{PH}$, compris entre 0 et 1, indique à quel point la machine "croit" que le prédicat $P$ correspond à la variable de tête $H$. Elle fait de même pour $\alpha_{P_1B_1}$ et $\alpha_{P_2B_2}$.

**Étape 3 : Calcul d'une déduction spécifique (Inférence)** La machine tente de déduire un nouveau fait $P$ à l'aide d'une règle $R$ et de deux faits de départ $P_1$ et $P_2$. Elle utilise l'**Équation suivante** pour calculer la probabilité de cette déduction :

- **$v(P,R,P_1,P_2) = (\alpha_{PH} \cdot \alpha_{P_1B_1} \cdot \alpha_{P_2B_2}) \cdot (v_{P_1} \cdot v_{P_2})$**.
- _Le premier crochet_ évalue si les prédicats épousent bien la forme de la règle.
- _Le second crochet_ est un "ET flou" (fuzzy AND) qui évalue si les faits de base utilisés sont bel et bien vrais.
- ==**_Le défaut majeur :_**== En multipliant autant de termes compris entre 0 et 1, le résultat chute. C'est le problème de **sous-estimation** (underestimation issue).

**Étape 4 : Recherche des meilleurs arguments** Puisqu'il existe d'innombrables faits pouvant théoriquement déclencher la règle, la machine ne conserve que la combinaison de faits $(P_1, P_2)$ qui lui donne le score de certitude le plus élevé pour la règle $R$ :

- **$v(P,R) = \max_{P_1,P_2} v(P,R,P_1,P_2)$**.

**Étape 5 : Mise à jour de la certitude globale** Il est possible que plusieurs règles différentes parviennent à prouver le même fait $P$. La machine effectue donc une disjonction (un "OU" logique traduit par un MAX) sur toutes les règles possibles, et met à jour sa croyance en la comparant avec ce qu'elle savait déjà ($v_{old}(P)$) :

- **$v(P) = \max \left(v_{old}(P), \max_R v(P, R) \right)$**.

**Étape 6 : L'apprentissage par l'erreur (Loss Function)**

- La machine répète ces calculs d'inférence (étapes 3 à 5) pendant $n_I$ étapes.
- À la fin, elle confronte ses certitudes finales $v(P)$ avec les vraies réponses attendues.
- Elle mesure son erreur à l'aide d'une fonction d'**Entropie Croisée Binaire (binary cross-entropy)**. Cette erreur lui permet de modifier les fameux vecteurs $\theta$ pour que ses prédictions soient meilleures au cycle suivant.