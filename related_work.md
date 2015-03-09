##PFID: Pittsburgh fast-food image dataset
###*ICIP'09*
- dataset:
	- 61 kinds of fast food with each contains 3x(4+4+6) images, 
	- two stereo images, one 360-degree video
	- train 2, test 1
	- use labortary images, 3-fold cross-validation
- baseline 1:
	- features: quantized color histogram
	- classifier: default libsvm (RBF kernel)
	- top-1 accuracy: 11.3%
- baseline 2:
	- features: bag of SIFT features
	- classifier: default libsvm (RBF kernel)
	- top-1 accuracy: 9.2%

-------------------------------------------------------------------------------
##A food image recognition system with Multiple Kernel Learning 
###*ICIP'09*
- dataset: 50-category with 100 images for each category
- 5-fold cross-validation
	- top-1 accuracy: 26.10% ~ 38.18% with single feature, 61.34% with MKL
- features
	- choose Point of interest by ramdom/grid-based/DoG
	- quantized color histogram of 2x2 blocks
	- Gabor filters bank -> average response
	- k-means + VQ for dictionary
	- bag of features
- classifier
	multiple kernel learning in SVM with &chi;2 kernel

-------------------------------------------------------------------------------
##Image processing based approach to food balance analysis for personal food logging
###*ICME'10*
- dataset
	- detect food: 93.3% accuracy (500 images)
	- balance estimation: 73% (with some bias catagory)
- online service
	- detect food: 85.3% in 500 images (92% after re-train, 8000 images)
	- balance estimation: 37% in 100 images(38% after re-train, 800 images)
	- personalized service can increase accuracy
- detecting features
	- circles detected by Hough transform
	- average value in color space
	- bag of features
- balance features
	- color histogram, DCT coefficients of 300 blocks
	- five groups of ingredient and non food (totally 6)
	- estimate servings (in number)
- classifier
	- both SVM in ingredient and food

-------------------------------------------------------------------------------
##Food Recognition Using Statistics of Pairwise Local Features
###*CVPR'10*
- dataset: PFID, 61 catagories with masked background
- 3-fold cross validation
	- train 2, test 1
	- top-1 for 61-catagory: range from 18.9% to 28.2%
	- top-1 for 7 major catagories: range from 69% to 78%
- features 
	- soft label 8-catagory ingredient using semantic texton forest
	- sampled pixels histogram of pairwise:  
	  distance/orientation/midpoint categories/between-pair categories/DO/OM
	- centered distance and orientation at mode
- classifier
	- SVM with &chi;2 kernel

-------------------------------------------------------------------------------
##Combining Global and Local Features for Food Identification in Dietary Assessment  
###*ICIP'11*
- dataset:
	- more than 1000 hand-segmented images under controlled condition
- user study: 
	- User 1: 19 catogories, 63 images
	- user 2: 28 catogories, 116 images
	- merge : 39 catogories, 179 images
- train 1, test 1
	- 10 times average
	- top-1: 98.1%, 97.2%, combined 86.1%
- global Features
	- 1st & 2nd moment of RGB, CbCr, ab, and HSV
	- RGB entropy for each segmented block -> average all blocks
	- predominant colors, percentage, color variance
	- Gabor energy -> mean, variance -> average all blocks
- local Features
	- local color of point of interest (POI)
	- local entropy within neighborhood of POI
	- local Gabor energy
	- some of Tamura features(coarseness, contrast, directionality)
	- SIFT 
	- Haar wavelet based on SURF
	- steerable filters -> 1st and 2nd moments
	- DAISY
- classifier
	- global: SVM with RBF kernel
	- local: 
		- DoG for POI -> local features -> hierarchical K-means -> 1NN

-------------------------------------------------------------------------------
##Automatic Chinese Food Identification and Quantity Estimation  
###*SIGGRAPH Asia 2012 Technical Briefs*
- 50 categories, 5000 images
- 5-fold cross validation
- training 3, fuse 1, validation 1
	- top-1: 68.3%, top-3: 84.8%, top-5: 90.9%
- quantity and nutrition estimation based on depth camera
- features
	- SIFT with sparse coding (as dictionary)
	- multi-resolution LBP with sparse coding
	- quantized RGB histogram
	- Gabor magnitudes -> mean and variance
- classifier
	- SVM for each features
	- multi-class AdaBoost with loss functino SAMME

-------------------------------------------------------------------------------
##A Novel SVM Based Food Recognition Method for Calorie Measurement Applications 
###*ICME'12 Workshops*
- dataset: 1636 images with 12 catagories of fruits and food
	- train 1, test 1
	- For each catagory top-1 accuracy range from 58.13% to 97.64% (avg. 92.6%)
- features 
	- segmentation
	- shape, texture, color, size
	- subtract food leftovers
- classifier
	- SVM

-------------------------------------------------------------------------------
##Segmentation and Recognition of Multi-Food Meal Images for Carbohydrate Counting 
###*BIBE'13*
- dataset: 
	- segmentation: HSI=0.885 in 65 images
	- recognition: >5000 images, >13000 patches
	- 10-fold cross-validation, top-1 accuracy: 87%
- features
	- convert to CIELAB
	- pyramidal mean-shift filtering (clustering, replaced by cluster's mean)
	- region growing (assign color)
	- region merging (merge region too small by k-means)
	- Plate Detection (RANSAC)/Background subtraction (outside plate)
	- histogram of the most dominant food colors (hierarchica k-means)
	- local binary pattern
- classifier
	SVM with RBF kernel

-------------------------------------------------------------------------------
##Food-101 ─ Mining Discriminative Components with Random Forests
###*ECCV'14*
- dataset: 101-catagory with 1000 images for each category 
	- 250 images were manually cleaned (testing)
	- train 3, test 1
	- top-1 accuracy: 50.76% (250 CPU hours)
	- Alexnet on this dataset: 56.40% (6 days using Tesla K20X)
- features
	- for each super pixel, extract: 
	  dense SURF in signed square root, Lab color space, 
	  encoded in Improved Fisher Vectors and GMM, 
	  and perform PCA-whitening
	- ramdomized generate some decision functions, 
	  choose one to maximize information gain on RF of superpixel until:
	  maximum depth, minimum samples, or single class remains
	- confidence score p(y|s): for each superpixel in validation set, 
	  average empirical distribution p(y|l) of reached leaves for each tree
	- distinctiveness(l|y): for each leaf, average reached samples' p(y|s),
	  sort by distinctiveness, filter out similar non-best-leaf
- classifier
	- for each leaf, train a linear binary SVM with hard negative mining
	- for each testing spatial region, uses 3-level spatial pyramid, 
	  average score of all superpixels within, 
	  and train a sructured-output multi-class svm

-------------------------------------------------------------------------------
##FoodCam: A Real-Time Mobile Food Recognition System Employing Fisher Vector
###*MultiMedia Modeling, 2014*
- dataset: 12905 images with 100 catagories
	- top-1: 51.9%, top-5: 79.2%
	- recognition time on Samsung Galaxy Note&#8545; costs 0.065s
- features
	- divide local patch into 2x2 blocks
	- extract (L2)normalized HOG (to unit) of eight orientations for each block
	- apply PCA from 32-dim to 24-dim
	- extract the mean and variance of RGB for each color patch, apply PCA
	- encode with Fisher vector, and apply power normalization with &alpha;=0.5
- classifier
	- linear SVM with one-versus-rest strategy

-------------------------------------------------------------------------------
##Multiple-food recognition considering co-occurrence employing manifold ranking
###*ICPR'12*

-------------------------------------------------------------------------------
##Food Detection and Recognition Using Convolutional Neural Network
###*MM'14*
