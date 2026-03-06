# Discours de Soutenance : Phase XGBoost

Ce document extrait la partie dédiée à l'eXtreme Gradient Boosting (XGBoost) issue du discours de soutenance global.

---

## 3. XGBoost

**Slide 29 : Sommaire - XGBoost**
« Nous traversons à présent le pont menant aux limites computationnelles modernes : l'eXtreme Gradient Boosting, plus communément XGBoost. »

**Slide 30 : Introduction à XGBoost**
« Si le Random Forest neutralise la variance par le principe du bagging, XGBoost est quant à lui taillé sur mesure pour s'attaquer au biais grâce au boosting. Introduite en 2016 par Tianqi Chen et Carlos Guestrin, cette architecture "eXtreme" repousse les limites de l'algorithme classique en optimisant non seulement la mathématique sous-jacente, mais aussi l'exécution matérielle. Extrêmement polyvalent pour des tâches de classification comme de régression, XGBoost s'est imposé comme le véritable standard de l'industrie sur les données tabulaires. Il est aujourd'hui célèbre pour rafler quasi-systématiquement les premières places lors des compétitions internationales de Data Science telles que Kaggle, s'affirmant comme l'alternative haute performance face aux méthodes traditionnelles d'agrégation. »

**Slide 31 : Caractéristiques et innovations de XGBoost**
« Avant de décortiquer son fonctionnement interne, attardons-nous sur les innovations techniques clés de XGBoost. Bien qu'il repose sur l'approche basique du Gradient Boosting, il va beaucoup plus loin. Mathématiquement, il intègre nativement une régularisation algébrique pour tuer le surapprentissage, un arbre de régression unique repensé, un algorithme glouton approché et une esquisse de quantiles pondérés pour trouver les coupures efficacement, même sur des données manquantes (le fameux "sparsity-aware"). Technologiquement, c'est une merveille d'ingénierie : il parallélise la construction au sein des nœuds, optimise l'accès au cache du processeur et utilise un système de blocs pour traiter de très vastes volumes de données qui ne tiendraient pas en mémoire vive, ce qu'on appelle "out-of-core computation". »

**Slide 32 : Principe du Gradient Boosting**
« Sous le couvercle de XGBoost se cache l'algorithme générique du Gradient Boosting. C'est un apprentissage itératif. À chaque étape temporellement dépendante de la précédente, le modèle suivant va ajuster son apprentissage en tentant de déduire, non pas la classification, mais formellement les "Erreurs", le résidu du modèle obsolète tout juste construit ! Sur le plan heuristique abstrait, il dresse une descente du gradient sur le plan formel des fonctions déductives, modulant ainsi continuellement la justesse. »

**Slides 33 et 34 : Construction séquentielle (Schémas)**
« Visuellement (Figure 6 et 7), on commence par un estimateur initial basique. Il dévie du réel (résidus). Le second arbre est spécialement mandaté pour "apprendre ces résidus". L'agglomération finale aboutit à une trajectoire prédictive chirurgicalement proche de la réalité. Chaque nouvel arbre, par la sommation ponctuelle de sa propre fonction de gain, permet de compenser doucement la défaillance des autres. »

**Slide 35 : Fonctionnalités spécifiques de XGBoost**
« Cependant, le Boosting ne suffit pas à définir XGBoost. XGBoost est "Extrême" à cause de son ingénierie informatique de génie. Il inclut un algorithme novateur appelé "Sparsity-aware split finding" qui attribue instantanément une direction de défaut à tout point numérique manquant au lieu de se bloquer ou de requérir de l'imputation artificielle. Il possède un subsampling par colonne ou au niveau du nœud (hérité de Random Forest !). Enfin, son moteur pénalise explicitement la longueur des feuilles de ses arbres par une régularisation algébrique très stricte. »

**Slides 36 et 37 : Subsampling et Early Stopping**
« (Figure 8 et 9) XGBoost emploie également le subsampling aléatoire pour combattre le surapprentissage inhérent au boosting. Et par-dessus tout, il gère l'Early Stopping interne (Arrêt prématuré) : dès l'instant ou la validation par la matrice annexe cesse de chiffrer en baisse pour la perte résiduelle au bout de N étapes redondantes, l'algorithme stoppe par sécurité la surchauffe inhérente à la prolifération arborisée. »

**Slide 38 : Régularisation**
« XGBoost intègre la pénalité Lasso de type absolue (nommée alpha L1) et la rigidité de crête (nommée lambda L2). (Figure 10) En somme, ces tensions bridées coupent drastiquement dans la force d'impact des résidus erratiques pour adoucir le contour géométrique global. »

**Slides 39 et 40 : Optimisations des systèmes & XGBoost vs GBM**
« Face au Gradient Boosting hérité historique (GBM), outre l’approximation d’ordre 2 via l’application vectorielle modélisée par la série mathématique de Taylor, l’architecture système est éminemment parallélisée par le classement préliminaire dans les cache mémoire du processeur en schéma vertical (Column Block), autorisant des sauts algorithmiques colossaux pour de vastes volumes. »

**Slide 41 : Histogram-based Split Finding**
« Plus spectaculaire (Figure 11), le Split des histogrammes. Traiter 1 million d'individus à chaque branche réclame des évaluations infinies (complexité $O(n \log n)$ au tri). XGBoost dérive par histogramme: on confine les décimales en bacs catégoriels figés de sous-intervalles. $O(n)$ baisse dramatiquement pour tomber en algorithme fixe discret indexé (vers l'ordre de grandeur de 256 paquets ou "bins"), propulsant ainsi la vitesse sans léser les performances cliniques. »

**Slide 42 : Hyperparamètres**
« Quant au tuning informatique, cet outil requiert une expertise. On configure la taille intrinsèque (max_depth), l'apprentissage temporel dit Learning_rate, conjugué à son corollaire absolu : l'early-stopping allié aux coefficients tensoriels "gamma" pour justifier légitimement les scissions. Sans oublier notre correcteur pour pathologies atypiques, le scale_pos_weight évitant toute dissimulation issue des déséquilibres ciblés. »

**Slide 43 : Vue radar XGB vs RF**
« En synthèse des modèles arborescents (Figure 12), si la Random Forest exécute massivement les tâches complexes par son agrégation en aveugle, XGBoost supplante systématiquement ses capacités prédictives dans n'importe quel arbitrage asymétrique nécessitant un ciblage microscopique de qualité à condition que le mathématicien l’opérant fournisse le maillage nécessaire des paramètres. »
