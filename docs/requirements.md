# Requirements — Camera arrière AR (stationnement)

## Objectif
Afficher le flux caméra arrière avec une superposition en réalité augmentée :
- courbes de trajectoire (guidelines) liées à l'angle de direction
- puis assistance au stationnement
- finalité long terme : manœuvre de stationnement automatisée (tests en environnement privé/fermé)

## Périmètre (Phase MVP)
- Capture vidéo temps réel
- Overlay (courbes + repères) stable
- Enregistrement vidéo pour rejouer/tester offline

## Hors périmètre (pour l’instant)
- Actionnement réel volant/frein/accélérateur
- Tests route ouverte
- “Full autonomous” en conditions réelles

## Contraintes
- Latence cible : < 150 ms (capture -> affichage)
- FPS cible : >= 20 fps
- Robustesse : overlay stable (pas de “glissement” visible)
- Sécurité : tests uniquement terrain privé/fermé, vitesse limitée, arrêt d’urgence prévu

## Entrées / sorties
Entrées :
- flux caméra arrière
- (plus tard) angle volant, vitesse véhicule, marche arrière engagée

Sorties :
- affichage vidéo + overlay
- logs + vidéos tests (hors git)

## Métriques de validation
- latence mesurée (timestamp)
- fps moyen
- test calibration : lignes droites corrigées (undistort)
- test guidelines : courbes cohérentes quand l’angle change
