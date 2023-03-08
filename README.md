# This repo is deprecated, please move to [new repo](https://github.com/ssocean/UNet-Binarization)

# Alleviating pseudo-touching in U-Net based binarization approach
code for <Zhao, P., Wang, W., Zhang, G., & Lu, Y. (2021). Alleviating pseudo-touching in attention U-Net-based binarization approach for the historical Tibetan document images. Neural Computing and Applications, 1-12.>
### Start with Train.py
python Train.py [imgs directory/] [masks directory/] (dont miss the last '/')
### Inference
python Prediction.py [imgs dir] [out_dir] [model_pth]
You are welcome to contact me anytime: oceanytech@gmail.com
### About Pseudo-touching  (假性粘连)
The optimal magnifications of various data might be different. We suggest trying various magnifications on your own dataset. 
