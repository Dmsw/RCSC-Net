1. Directory structure
```
\saved_models
    \cave
        \beta_2d_pt       pretrained model
        \gamma_ftall      finetuned model

pretrain_cscnet.py        pretraining the 2D CSC-Net on dataset with Clean-Noisy Paired
RCSCNet.py                Model
train_rcscnet.py          Finetuning the 3D RCSC-Net on target Noisy HSI

```


2. Results on Real World Dataset
Results on two Real World Dataset
2.1. Indian 
Input image at 149th band
![Input](images/noise5.png){:height="256px" width="256px"}
Reconstructed image at 104th band by BM3D
![Ours](images/bm3d5.png){:height="256px" width="256px"}
Reconstructed image at 104th band by BM4D
![Input](images/bm4d5.png){:height="256px" width="256px"}
Reconstructed image at 104th band by LRTV
![Ours](images/lrtv5.png){:height="256px" width="256px"}
Reconstructed image at 104th band by SIP
![Input](images/sip5.png){:height="256px" width="256px"}
Reconstructed image at 104th band by HSI-TS
![Ours](images/hsits5.png){:height="256px" width="256px"}
Reconstructed image at 104th band by N2N
![Input](images/nei2nei5.png){:height="256px" width="256px"}
Reconstructed image at 104th band by Ours
![Ours](images/ours5.png){:height="256px" width="256px"}
2.2. Urban 
Input image at 104th band
![Input](images/noise6.png){:height="256px" width="256px"}
Reconstructed image at 104th band by BM3D
![Ours](images/bm3d6.png){:height="256px" width="256px"}
Reconstructed image at 104th band by BM4D
![Input](images/bm4d6.png){:height="256px" width="256px"}
Reconstructed image at 104th band by LRTV
![Ours](images/lrtv6.png){:height="256px" width="256px"}
Reconstructed image at 104th band by SIP
![Input](images/sip6.png){:height="256px" width="256px"}
Reconstructed image at 104th band by HSI-TS
![Ours](images/hsits6.png){:height="256px" width="256px"}
Reconstructed image at 104th band by N2N
![Input](images/nei2nei6.png){:height="256px" width="256px"}
Reconstructed image at 104th band by Ours
![Ours](images/ours6.png){:height="256px" width="256px"}

1. Convergence Analysis
The changes in PSNR and loss across training epochs are visualized in the plot for CAVE case 3.
![Convergence](images/epoch.jpg)
Generally, our model converged with epochs of 200.
