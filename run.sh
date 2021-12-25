python inference_gfpgan.py \
	--upscale 2 \
	--test_path inputs/upload \
	--save_root results \
	--model_path experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth \
	--bg_upsampler realesrgan
