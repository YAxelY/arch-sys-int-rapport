# Discours de Soutenance : Support Vector Machines, Random Forests & Extreme Gradient Boosting

Ce document propose un discours de présentation complet, détaillé et structuré slide par slide, basé à la fois sur le diaporama et sur l'analyse approfondie du rapport final. Ce discours s'adresse à un public académique de niveau Master 2 en Intelligence Artificielle.

---

## 1. Introduction

**Slide 1 : Page de Garde**
« Bonjour à toutes et à tous, et merci au jury de nous recevoir aujourd'hui. Nous sommes le groupe 3, composé de Vergez, Aubin, Hamed et Senge. Nous avons le plaisir de vous présenter aujourd'hui notre étude comparative portant sur trois algorithmes phares de l'apprentissage automatique : les Support Vector Machines, les Random Forests et l'Extreme Gradient Boosting, appliqués au domaine de la santé. »

**Slides 2 à 4 : Sommaire**
*(Passage rapide sur le sommaire)*
« Notre présentation s'articulera en quatre temps forts. Nous débuterons par une introduction aux concepts clés de l'apprentissage supervisé, puis nous plongerons dans la mécanique du bagging avec les Random Forests. Nous étudierons ensuite l'approche séquentielle du boosting avec XGBoost, avant de détailler la théorie géométrique des SVM. Enfin, nous conclurons par une synthèse comparative étayée par notre cas pratique d'application clinique. »

**Slide 5 : Introduction**
« Pour contextualiser, le machine learning supervisé regorge d'algorithmes puissants, mais les SVM, les Random Forests et XGBoost se distinguent nettement comme les piliers de l'état de l'art pour les données tabulaires. Chacun adopte un paradigme radically différent : là où le SVM cherche l'hyperplan optimal séparant des points dans l'espace, le Random Forest parie sur la sagesse d'une foule d'arbres indépendants, et XGBoost corrige itérativement les erreurs de ses prédécesseurs. L'objectif de notre étude est de comprendre ces mécanismes profonds. »

**Slide 6 : Concepts clés : Apprentissage Automatique**
« Rappelons d'abord des concepts essentiels. Le Machine Learning permet aux systèmes d'extraire des modèles à partir des données sans programmation explicite. Dans le cadre supervisé, que nous traitons, nous distinguons deux tâches : la classification, où l'on cherche à prédire une étiquette discrète (comme sain ou malade), et la régression, pour prédire une valeur continue. Notre cas d'application ultérieur se concentrera sur la classification binaire. »

**Slide 7 : Concepts clés : Bagging vs Boosting**
« Avant d'entrer dans le vif du sujet, il est crucial de différencier les deux méta-architectures ensemblistes. À gauche sur la figure, le Bagging (ou Bootstrap Aggregating) consiste à entraîner plusieurs modèles indépendants en parallèle sur différents sous-échantillons. Le but y est de réduire la variance ; c'est le principe des Random Forests. À droite, le Boosting est une méthode séquentielle où chaque nouveau modèle tente de corriger les erreurs résiduelles des modèles précédents. Le but prioritaire ici est de réduire le biais d'estimation ; c'est le moteur de XGBoost. »

---

## 2. Random Forests

**Slide 8 : Sommaire - Random Forests**
« Plongeons à présent dans la mécanique des forêts aléatoires, ou Random Forests. »

**Slide 9 : Introduction aux Random Forests**
« Historiquement, un arbre de décision isolé est instable et très sujet au surapprentissage. Les Random Forests, formalisées par Leo Breiman, résolvent ce problème en construisant une multitude d'arbres. Le principe repose sur un théorème intuitif : la sagesse des foules. En combinant les décisions d'apprenants dits "faibles", nous formons un algorithme "fort" très résistant au bruit. À la différence de XGBoost, ces arbres sont construits de manière totalement indépendante et en parallèle. »

**Slide 10 : Fonctionnement du Random Forest (Schéma)**
« Comme illustré sur ce schéma, le dataset principal est scindé en multiples sous-échantillons. Chaque sous-échantillon permet d'entraîner un arbre spécifique. Pour une nouvelle prédiction, la donnée traverse tous les arbres de la forêt. Leurs prédictions finales sont ensuite agrégées par vote majoritaire pour donner le résultat final. »

**Slide 11 : Rappel : Arbres de décision**
« Faisons un bref rappel sur la structure de base : l'arbre de décision. Il s'agit d'un séquençage hiérarchique de questions binaires (oui ou non). Chaque nœud interne opère un clivage d'une variable. Les branches sont les chemins pris en fonction de la réponse, et la feuille finale représente la classification du modèle. »

**Slide 12 : Rappel visuel : Arbre de décision**
« Sur cet exemple classique, on observe la ramification simple basée sur l'Âge et le Revenu. L'algorithme cherche à maximiser le gain d'information à chaque nœud, isolant progressivement les classes de manière pure. »

**Slide 13 : Construction récursive et surapprentissage**
« L'inconvénient majeur de cette croissance purement récursive est qu'un arbre poussé à sa profondeur maximale finira par mémoriser par cœur les bruits du jeu d'entraînement. Il aura un biais très faible, mais une variance extrêmement élevée. La Random Forest va attaquer directement cette variance. »

**Slide 14 : Algorithme des Random Forests**
« Comment s'y prend-elle ? L'algorithme introduit une double randomisation ou stochastisation à la création des arbres : le tirage Bootstrap des données, et surtout, la sélection aléatoire des sous-ensembles de caractéristiques (features) à chaque nœud. »

**Slide 15 : Étape 1 : Échantillonnage Bootstrap**
« La première étape est le Bootstrap. Pour construire un arbre, on pioche aléatoirement "N" observations dans le dataset d'origine, mais avec remise. Cela signifie que mathématiquement, un échantillon Bootstrap ne contiendra en moyenne que 63,2% d'observations uniques de la base. Le reste est constitué de doublons. Cette première randomisation assure que chaque arbre apprend d'une perspective légèrement différente. »

**Slide 16 : Étape 2 : Sélection aléatoire des caractéristiques**
« La deuxième étape, c'est l'innovation majeure de Breiman : la division stochastique des nœuds. Au moment de créer un nœud de décision, au lieu de tester toutes les variables disponibles, l'algorithme n'évalue qu'un petit sous-ensemble de variables tiré au hasard (par exemple la racine carrée du nombre total). Cela force la forêt à utiliser des variables moins dominantes et casse la corrélation structurelle entre les arbres. C'est ce qui rend la forêt véritablement résiliente. »

**Slide 17 : Étapes 3 et 4 : Croissance et Agrégation**
« La croissance des arbres se fait alors sans aucun élagage. Puis vient la prédiction. Dans le cas d'une classification, l'agrégation se fait sagement au vote majoritaire. S'il y a 500 arbres et que 350 votent "Diabétique", le modèle choisira cette classe. Pour une régression, ce sera la moyenne pure. »

**Slide 18 : Hyperparamètres : Nombre d'arbres**
« Côté optimisation, le nombre d'arbres (n_estimators) dicte l'étendue de la forêt. Contrairement à beaucoup de modèles, ajouter des arbres à un Random Forest ne provoque pas mathématiquement de surapprentissage. Les performances finissent juste par atteindre un palier asymptotique. En général, 100 à 500 arbres suffisent. »

**Slide 19 : Hyperparamètres : max_features**
« Toutefois, l'hyperparamètre le plus névralgique est le "max_features", qui contrôle la taille du sous-ensemble de descripteurs à évaluer par nœud. En classification, on retient souvent la racine carrée du nombre de variables ; en régression, le tiers. Cette limite de détection règle finement l'équilibre entre la puissance de chaque arbre et la diversité globale de la forêt. »

**Slide 20 : Autres hyperparamètres et stratégie**
« En pratique, la profondeur maximale (max_depth) ou le seuil de présence par nœud foliaire limitent encore le surapprentissage. Pour optimiser, une stratégie saine consiste à ajuster le contingent d'arbres d'abord, puis à s'intéresser au "max_features", en suivant son comportement naturel vis-à-vis des données test via l'OOB. »

**Slides 21 à 23 : Erreur Out-of-Bag (OOB)**
« Parlons justement de l'OOB, l'erreur Out-Of-Bag. C'est un outil fascinant spécifique au bagging. Lors du tirage avec remise, 36,8% des données ne sont pas vues par l'arbre. Elles peuvent donc servir instantanément de jeu de validation interne "gratuit" pour cet arbre précis. L'erreur OOB, obtenue en moyennant les erreurs sur ces données invisibles, s'avère être une approximation très fidèle, et non biaisée, de la capacité de généralisation terminale du modèle. Nul besoin de valider par un subset exclusif si l'échantillon est restreint ! »

**Slides 24 et 25 : Importance des variables (MDI)**
« Un autre bénéfice capital des Random Forests est la fourniture "gratuite" de l'importance des variables. Via la méthode de la diminution moyenne de l'impureté de Gini (MDI), nous traçons le poids relatif imparti à l'attribut à chaque bifurcation. Même si cette méthode de Gini tend parfois à privilégier les attributs continus ou à forte cardinalité face aux facteurs discrets, elle donne une cartographie phénoménale du paysage des informations discriminatoires. »

**Slides 26 à 28 : Conclusion Random Forests**
« Pour synthétiser sur la forêt aléatoire : on plébiscite sa robustesse prodigieuse à des échantillons atypiques et son exigence d'un prétraitement modeste. Elle excelle par sa capacité innée de parallélisation matricielle. Ses faiblesses se localiseront principalement au niveau du manque d'extrapolation pure en valeur absolue dans les cas de régressions, ainsi qu'une apparente lourdeur binaire pour l'interprétariat comparée à un unique arbre direct. Mais retenez ceci : la forêt aléatoire dompte la variance par son fonctionnement simultané et autonome. À présent, abordons la logique inverse avec XGBoost. »

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

**Slides 32 et 33 : Construction séquentielle (Schémas)**
« Visuellement (Figure 6 et 7), on commence par un estimateur initial basique. Il dévie du réel (résidus). Le second arbre est spécialement mandaté pour "apprendre ces résidus". L'agglomération finale aboutit à une trajectoire prédictive chirurgicalement proche de la réalité. Chaque nouvel arbre, par la sommation ponctuelle de sa propre fonction de gain, permet de compenser doucement la défaillance des autres. »

**Slide 34 : Fonctionnalités spécifiques de XGBoost**
« Cependant, le Boosting ne suffit pas à définir XGBoost. XGBoost est "Extrême" à cause de son ingénierie informatique de génie. Il inclut un algorithme novateur appelé "Sparsity-aware split finding" qui attribue instantanément une direction de défaut à tout point numérique manquant au lieu de se bloquer ou de requérir de l'imputation artificielle. Il possède un subsampling par colonne ou au niveau du nœud (hérité de Random Forest !). Enfin, son moteur pénalise explicitement la longueur des feuilles de ses arbres par une régularisation algébrique très stricte. »

**Slides 35 et 36 : Subsampling et Early Stopping**
« (Figure 8 et 9) XGBoost emploie également le subsampling aléatoire pour combattre le surapprentissage inhérent au boosting. Et par-dessus tout, il gère l'Early Stopping interne (Arrêt prématuré) : dès l'instant ou la validation par la matrice annexe cesse de chiffrer en baisse pour la perte résiduelle au bout de N étapes redondantes, l'algorithme stoppe par sécurité la surchauffe inhérente à la prolifération arborisée. »

**Slide 37 : Régularisation**
« XGBoost intègre la pénalité Lasso de type absolue (nommée alpha L1) et la rigidité de crête (nommée lambda L2). (Figure 10) En somme, ces tensions bridées coupent drastiquement dans la force d'impact des résidus erratiques pour adoucir le contour géométrique global. »

**Slides 38 et 39 : Optimisations des systèmes & XGBoost vs GBM**
« Face au Gradient Boosting hérité historique (GBM), outre l’approximation d’ordre 2 via l’application vectorielle modélisée par la série mathématique de Taylor, l’architecture système est éminemment parallélisée par le classement préliminaire dans les cache mémoire du processeur en schéma vertical (Column Block), autorisant des sauts algorithmiques colossaux pour de vastes volumes. »

**Slide 40 : Histogram-based Split Finding**
« Plus spectaculaire (Figure 11), le Split des histogrammes. Traiter 1 million d'individus à chaque branche réclame des évaluations infinies (complexité $O(n \log n)$ au tri). XGBoost dérive par histogramme: on confine les décimales en bacs catégoriels figés de sous-intervalles. $O(n)$ baisse dramatiquement pour tomber en algorithme fixe discret indexé (vers l'ordre de grandeur de 256 paquets ou "bins"), propulsant ainsi la vitesse sans léser les performances cliniques. »

**Slide 41 : Hyperparamètres**
« Quant au tuning informatique, cet outil requiert une expertise. On configure la taille intrinsèque (max_depth), l'apprentissage temporel dit Learning_rate, conjugué à son corollaire absolu : l'early-stopping allié aux coefficients tensoriels "gamma" pour justifier légitimement les scissions. Sans oublier notre correcteur pour pathologies atypiques, le scale_pos_weight évitant toute dissimulation issue des déséquilibres ciblés. »

**Slide 42 : Vue radar XGB vs RF**
« En synthèse des modèles arborescents (Figure 12), si la Random Forest exécute massivement les tâches complexes par son agrégation en aveugle, XGBoost supplante systématiquement ses capacités prédictives dans n'importe quel arbitrage asymétrique nécessitant un ciblage microscopique de qualité à condition que le mathématicien l’opérant fournisse le maillage nécessaire des paramètres. »

---

## 4. SVM

**Slide 43 : Sommaire - SVM**
« Changeons brutalement de paradigme géométrique. Adieu les arbres et leurs fractionnements rectangulaires, plongeons au cœur de l'algèbre bilinéaire des Support Vector Machines, les SVM. »

**Slide 44 : Support Vector Machines (Définition)**
« Pionniers des années 90, portés par le mathématicien Vapnik, les Séparateurs à Vaste Marge conçoivent l'échantillon sous forme de vecteurs dispersés dans l'espace multidimensionnel. L'assise centrale se targue de dénicher l'infiniment unique plan séparateur - "l'Hyperplan optimal". Son identité : maximiser au mieux l'espace absolu de vide ou "Marge" qui le distance du membre majoritaire et minoritaire le plus proche de la ligne de touche. Ces points à haut risque sont nommés "Vecteurs de support". »

**Slides 45 à 47 : Intuition graphique**
« (Slides 13 et 14). Imagions cette métaphore. Soit un domaine de dispersion de formes de couleurs. Placer une limite franche semble simple, or la question centrale est : Quelle est la meilleure ligne ? Celle offrant la garantie absolue de la généralisation aux nouveaux triangles (futurs carrés ou ronds) ne se colle à personne, elle divise l'air au point d'équité la plus majestueuse possible. »

**Slides 48 à 50 : Formalisme et Marge**
« Abordons le squelette formel. L'équation primitive pose l'hyperplan par la fonction $h(x) = W^\top X + b$. W symbolise l'inclinaison orthogonale de notre vecteur séparateur, et "b" sa variation constante. Avec ce système posé, on borne par l'intolérable toute mauvaise classification : la multiplication de la consigne (Positive/Négative) face au plan doit légitimement demeurer supérieure à l'unité symbolique (la barrière de la Marge !).
Le joyau mathématique est que "la distance" pure encadrée dans ces marges équivaut algébriquement à $\frac{2}{\|W\|}$. Par inversion stricte, agrandir cet espace stérile de séparation (la Marge maximale !) stipule que notre apprentissage revient exclusivement à "Minimiser la dimensionnelle de notre tenseur $W$". Le cadre d'optimisation lagrangien vient d'être débusqué. »

**Slide 51 : Formulation Lagrangienne**
« Assorti aux multiplicateurs de restriction, la descente aboutit à la formalisation dite "du problème dual par alpha ($a$)". Une conclusion époustouflante s'impose brutalement : sur plusieurs milliers de constantes d'observations, l'immense fraction du volume prend la pondération rigoureusement mathématique "Zéro". Seuls un groupuscule intime de vecteurs frôlant des seuils de la limite (Alpha $\neq 0$) existera dorénavant, incarnant intrinsèquement, à eux seuls, notre modèle prédictif total. Nos Vrais Vecteurs de Support existent. »

**Slides 52 à 55 : SVM Non Linéaire & Astuce Plongée**
« Devant l'incapacité d'une droite droite à séparer deux cercles s'emboîtant, la restriction linéaire fléchit (slide 16). La projection du plongement pallie à cette erreur. Si l'on augmente virtuellement les dimensions en attribuant via la fonction Phi $ϕ()$ des puissances carrées supplémentaires, les formes isolées originelles se lèvent, ou s'assurent un relief altéré sur l'axe Z inédit (graphe de la figure 18). L'entremêlement plan insoluble accouche d'une fracture rectiligne aisée un cran dimentionnellement plus haut. Puis en basculant algébriquement dans le premier plan originel (la redescente), cette nappe mathématique rigide engendre implicitement la création miraculeuse d'un encerclement sphérique purement non-linéaire sur les points de la carte mère originelle. »

**Slide 56 et 57 : Le Kernel Trick (Types de noyaux)**
« Cependant cette projection est astronomiquement chère : elle alourdit le nombre d'entrées. Intervient alors le miracle fonctionnel du Kernel Trick ("l'Astuce du Noyau"). En vertu du fameux produit scalaire dualiste, si la méthode remplace formellement l'association formelle matricielle du plongement par l'application d'un Noyau à matrice de covariance prédéfinie (exemple du noyau RBF Radial Gaussien), nous accédons directement à ce degré de courbure supérieur... au bas prix de la dimension du dataset originel ! Nul besoin d'exploser littéralement les entrées du calculateur, le coulage s'articule par magie formelle implicite. »

**Slide 58 : Hyperparamètres SVM**
« Finalement on équilibre cette merveille avec ses Hyperparamètres. Soit C (assouplissement marginial dit régularisateur L1 lissée). Soit le Gamma de l'inclinaison des sommets RBF : une propagation lissée ou infiniment segmentée en piquet. Ces deux réglages se modèlent via Grid-Search combinatoire. Ils déterminent avec autorité l'arbitrage "Surapprentissage face à tolérance du modèle global". »

**Slide 59 : Conclusion SVM**
« La synthèse de l'hyperplan vectoriel s'inscrit en un acronyme : une solution garantie du point mathématique sur les calculs géométriques stricts, mais intimement sclérosée devant d'abondants gisements de big-data truffés d'erreurs non asymétriques où l'assemblage décisionnel arborescent fait loi. »

---

## 5. Conclusion Générale

**Slide 60 : Sommaire - Conclusion Générale**
« Nous approchons du point d'aboutissement de notre étude. »

**Slide 61 : Tableau récapitulatif théorique**
« L'inspection des paradigmes nous livre un tableau comparatif sans faille. Le SVM incarne l'orthodoxie spatiale du séparateur maximal. Le Random Forest s'adjuge l'audace agglomérée des contingences indépendantes. Et XGBoost orchestre le ballet implacable et ciblé de la séquence rectifiée des faiblesses préexistantes via Gradient de 2ème ordre. L'exigence de configuration diffère grandement entre le pragmatisme simplissime des forêts aléatoires et la surabondance systémique de boosting gradient extrême. »

**Slide 62 : Cas d'étude - Détection du diabète (PIDD)**
« Ces algorithmes n'auraient de finalité sans un banc d'essai exhaustif. Nous avons orchestré l'ensemble de ces outils de pointe au service clinique en confrontant leurs forces face au Dataset très particulier Pima Indians Diabetes, dont les 768 patientes et le grand facteur modérateur du glucose constituaient notre base vectorielle. Outillé d'une refonte hybride structurelle (imputation, Borderline SMOTE pour la minorité systémique, optimisation poussée Optuna sur XGBoost), nous avons quantifié l'architecture.
La dominance de XGBoost fut irrévocable, captant l'harmonie par un Area-Under-The-Curve stratosphérique fixé à 0.9496 à lui seul. Random forest consolide magistralement, tandis que l'architecture SVM, souffrant d'effondrement paramétrique en zone tabulaire imbriquée, avoua un index modeste à 0.8589 avec des sensibilités chroniques. La modélisation médicale à très hauts enjeux est le domaine sacré des gradients asymétriques pénalisés. »

**Slide 63 : The End**
« La force de compréhension de la donnée pré-filtrée compte en tout domaine autant que le triomphe paramétrique de la classification choisie. Nous vous remercions chaleureusement pour votre écoute et votre attention bienveillante, et nous nous tenons à votre entière disposition pour répondre à vos éventuelles questions. »
