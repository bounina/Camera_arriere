# Test plan

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
