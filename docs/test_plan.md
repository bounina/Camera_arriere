# Test Plan

## Objectif
Valider étape par étape sans danger : d’abord le pipeline vidéo, puis calibration, puis guidelines dynamiques, puis simulation.

## Phase 1 — Smoke test vidéo
- Test : ouvrir caméra / lire une vidéo
- Attendu : affichage fluide, pas de crash
- Mesures : fps + latence (approx)

## Phase 2 — Calibration / Undistort
- Test : afficher image brute vs undistort
- Attendu : distorsion réduite (lignes droites mieux alignées)
- Artefacts : sauvegarder une capture avant/après (dans un dossier local non versionné)

## Phase 3 — Guidelines dynamiques
- Test : simuler angle volant (valeurs fixes) et vérifier que les courbes changent
- Attendu : forme cohérente, pas de tremblement, overlay stable

## Phase 4 — Replay sur enregistrements
- Test : rejouer une vidéo enregistrée + appliquer overlay
- Attendu : même rendu, reproductible
- But : permettre tests offline (CI plus tard)

## Sécurité (si tests réels)
- Terrain privé/fermé
- vitesse limitée
- arrêt d’urgence accessible
- un opérateur dédié à la sécurité
