# Test plan

## Phase 0 – Camera Smoke Test

Objectif: valider une base caméra "couleurs justes" avant tout overlay/courbe.

### Commandes de validation

1. Local (écran Pi):
   - `python3 tools/camera_smoke_test.py`
2. VNC (fenêtre plus légère):
   - `python3 tools/camera_smoke_test.py --display-scale 0.5`
3. Dump diagnostic 1ère frame:
   - `python3 tools/camera_smoke_test.py --dump-first-frame --headless --headless-frames 5`
4. Headless sans DISPLAY (ex: SSH pur):
   - `DISPLAY= python3 tools/camera_smoke_test.py --headless --headless-frames 5`

### Critères de validation

- Le flux est visible (ou des images sont écrites en headless) avec un HUD minimal: FPS, résolution, format annoncé et shape réel.
- La première frame loggue `raw_shape`, `raw_dtype` et la conversion appliquée (source de vérité = frame réelle, pas seulement le format annoncé).
- `--dump-first-frame` crée:
  - `data/screenshots/raw.npy` (frame brute)
  - `data/screenshots/bgr.png` (frame convertie BGR)
- Critère principal couleur: `bgr.png` doit correspondre visuellement à `rpicam-hello` (peau naturelle, pas de dominante violette).
- Touches: `q` quitte, `s` sauvegarde un screenshot BGR.
- Ctrl+C quitte proprement sans stacktrace; la caméra est stoppée et les fenêtres sont fermées.

## Phase 1 preview + overlay

1. Lancer `python3 tools/monscript.py` sur Raspberry Pi avec la caméra CSI branchée.
2. Vérifier que la fenêtre s'ouvre et que le flux est fluide.
3. Vérifier l'overlay: 2 lignes fixes + 1 courbe centrale visibles.
4. Vérifier les infos overlay: FPS, résolution, steering.
5. Appuyer sur `a` et `d`: la courbe doit tourner gauche/droite.
6. Appuyer sur `r`: l'angle revient à `0`.
7. Appuyer sur `s`: une capture est créée dans `data/screenshots/`.
8. Appuyer sur `q`: fermeture propre sans crash.
9. (Optionnel SSH) Lancer `python3 tools/monscript.py --headless --headless-frames 30` et vérifier l'écriture de captures de debug.
