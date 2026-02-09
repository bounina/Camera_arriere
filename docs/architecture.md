# Architecture

## Modules (cibles)
1) Video I/O
- capture caméra + timestamps
- option : lecture vidéo enregistrée (replay)

2) Calibration
- intrinsics + distortion (fisheye si besoin)
- undistort temps réel

3) Vehicle model (guidelines)
- génération des courbes de trajectoire à partir de l’angle de direction + empattement

4) Perception (plus tard)
- détection place/lignes / marqueurs (ex: ArUco pour zone test)

5) Planning & Control (simulation d’abord)
- planification trajectoire de parking
- contrôle (simulateur) puis réel (très bridé)

6) UI Overlay
- rendu des courbes
- affichage debug (fps, latence, angle)

## Arborescence
- docs/ : specs, architecture, plan de test, ADR
- src/ : code principal
- configs/ : calibration caméra, paramètres véhicule
- data/ : README uniquement (pas de vidéos lourdes dans git)
- tests/ : tests unitaires + tests sur enregistrements
- tools/ : scripts (capture, conversion, replay)

## Formats de config (proposition)
- configs/camera.yaml : calibration + résolution
- configs/vehicle.yaml : empattement, largeur, limites angle
