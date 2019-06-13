
GAN-Generate_Cartoon_image
=

env:
-
* cpu/gpu

* python3.6

* torch1.1

* requirements.txt

train:
    python main.py train --gpu 

Generate pic command:

    python main.py generate --nogpu --vis=False --netd-path=checkpoints/netd_3.pth --netg-path=checkpoints/netg_3.pth 
    
result
-
![pic]()