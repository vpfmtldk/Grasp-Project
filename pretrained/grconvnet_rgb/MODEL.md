# Final grasp model -- GR-ConvNet, RGB-only

**THE model.** Downstream (experiment 2, real-photo prediction, robot execution)
loads `output/models/final_grconvnet_rgb1_d0/weights.pt`. Do not retrain.

## Build
- Arch: GR-ConvNet (models/grconvnet.py), input_channels=3 (RGB), no depth
- Data: Cornell Grasping Dataset (885 img), trained on split=0.95, ds-rotate=0
- 300x300 centre crop, 25 epochs, Adam (default LR), batch 8
- Command: run_train.ps1 -Network grconvnet -UseRgb 1 -UseDepth 0 -Split 0.95 -Epochs 25 -RunDir output/models/final_grconvnet_rgb1_d0
- Selected checkpoint: epoch_22 (best held-out IoU on the 5% val split)

## Expected performance -- 5-fold image-wise CV (25 epochs each)
mean IoU = 0.903, sd = 0.054   (folds: 0.944 / 0.831 / 0.843 / 0.944 / 0.955)
RGB-only, no depth (matches the monocular camera rig).

## Files
- weights.pt : state_dict; load via eval_ggcnn.load_network (arch auto-detected) or GRConvNet(input_channels=3)
- model.pt   : full pickled module
