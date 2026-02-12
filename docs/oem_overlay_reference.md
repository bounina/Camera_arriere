# OEM Parking Overlay Reference (Phase 2)

## Éléments visuels d'un overlay OEM

Un overlay OEM de caméra de recul combine généralement une zone de guidage correspondant à la **largeur véhicule**, des **repères de distance** (proximité pare-chocs) et des **lignes dynamiques** qui suivent la trajectoire prévue. Les deux lignes fixes latérales définissent le couloir de roulage arrière immédiat. Les repères (ex. 0.5m, 1m, 2m, 3m) donnent un ordre de grandeur de distance avant obstacle. Les lignes dynamiques indiquent la trajectoire attendue pour l'angle de braquage courant, afin d'anticiper l'emprise du véhicule en marche arrière. Le style visuel est souvent fort contraste (jaune/blanc/vert), lisible en plein jour et de nuit.

## Lignes dynamiques: principe de calcul (Ackermann / bicycle)

1. On modélise le véhicule avec un modèle bicycle (un essieu avant directeur, un essieu arrière suiveur).
2. L'entrée est l'angle de volant (ou angle de roue équivalent) noté `delta`.
3. La courbure de trajectoire est liée au rayon instantané `R ≈ L / tan(delta)` avec `L` l'empattement.
4. Si `delta = 0`, la trajectoire est quasi droite (rayon infini).
5. Si `delta > 0`, la trajectoire tourne d'un côté, si `< 0` de l'autre.
6. On échantillonne des points futurs de la trajectoire sur quelques mètres.
7. Chaque point est exprimé dans le repère sol du véhicule (x latéral, y longitudinal).
8. En Phase 2A, on trace encore en pixels (approximation visuelle).
9. Ensuite on projette ces points sol vers l'image via homographie/caméra calibrée.
10. Cette projection plan-sol rend les repères cohérents en mètres et robustes selon la perspective.

## Phases

- **2A – Pixels:** overlay OEM en coordonnées image, commandes de braquage simulé, validation UX et lisibilité.
- **2B – Mètres + homography:** passage à une géométrie physique (plan sol), projection cohérente en image.
- **3 – Calibration:** calibration caméra/véhicule (intrinsèques/extrinsèques), alignement précis des guides.
