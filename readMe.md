3D-CNN

  3D-CNN is a CNN framework based on [darknet](https://github.com/pjreddie/darknet)
  
  Usage:
  
        $ sudo apt-get install p7zip
	$ git clone https://github.com/yp-wang/3d-cnn
	$ cd 3d-cnn
	$ 7z x 3dNetworkDataset.7z -r -o .
	$ make
	$ ./3DNet [train/valid] [cfg] [weights (optional)]
