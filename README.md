#Master Thesis Proposal
--------------------------------------------------------------------------------
##Current Result
- color histogram(by SIGGRAPH'12)
	- linear SVM: 26.94% (c=0.1, need to tune parameter)
	- RBF SVM: 30.5%(c=8, g=0.125)
- raw image classified by RBF-SVM: 26.92%
    - tuned by libsvm/tools/grid.py
    - best parameter: 
        - c=2<sup>3</sup>, g=2<sup>-15</sup>, rate=26.92% (5-fold)
- raw image classified by linear-SVM: 18% (14.52% with -c 0.01)
    - recall rate = 94%
    - severe overfitting, parameter-tuning needed
- bvlc\_alexnet: 47%
    - [Slide](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf)
    - [Paper](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
    - current iteration: 24000 (max. 450000)
    - **STILL RUNNING**

-------------------------------------------------------------------------------
##Preliminary Method: 
###(Sparse Coding) Spatial Pyramid Matching(Implementing)
- Divide image from coarse to fine-grain
- Extract features, compare similarity with grid penalty (course > fine-grain)
- (Possible) Combine sparse coding for using linear kernel
- (Possible) Use locality-constrain linear coding(LLC) to substitute sparse coding
	- locality is important than sparsity (according to CVPR'10)

###Hard-Labeled Patch with Majority Vote (Na&iuml;ve)
- In training phase, divide training image into patches with label,
  then learn a classifier to hard-classify patch. 
- For testing, divide testing image into patches,
  and classify the image with majority label of patches.

------------------------------------------------------------------------------
##Idea
- convolution neural network
    - need to survey network design
- hand-crafted features with SVM

-------------------------------------------------------------------------------
##Problem
- No baseline source code, I need to implement from scratch
- Baseline CNN is slow, I'm afraid it may not converge in time.
- GTX 970 ?

------------------------------------------------------------------------------
##Related Work
[Press Me](related_work.md)

-------------------------------------------------------------------------------
##Motivation
Daily diet gains more and more attention nowaday. 
However, Most of automatic food image identification systems still 
suffer from overfitting and low accuracy, 
while self-report-based systems need extra care to get regular record. 
Thus, an accurate automatic food-log system is needed.

-------------------------------------------------------------------------------
##Problem Description
Given a dataset with labeled food images, 
propose a system to identify food catagory of unknown images, 
and evaluate proposed system with top-1 (and top-5 accuracy, if possible).

-------------------------------------------------------------------------------
##Expected Result
- Achieve 50% ~ 65% of top-1 accuracy on "50data" or "food101"
- Reduce the classification computational complextiy to acceptable level.

-------------------------------------------------------------------------------
##Schedule
- 03/08 - 03/14 (week 03) : 
- 03/15 - 03/21 (week 04) : 
- 03/22 - 03/28 (week 05) : 
- 03/29 - 04/04 (week 06) : 
- 04/05 - 04/11 (week 07) : 
- 04/12 - 04/18 (week 08) : 
- 04/19 - 04/25 (week 09) : 
- 04/26 - 05/02 (week 10) : 
- 05/03 - 05/09 (week 11) : 
- 05/10 - 05/16 (week 12) : 
- 05/17 - 05/23 (week 13) : 
- 05/24 - 05/30 (week 14) : 
- 05/31 - 06/06 (week 15) : 
- 06/07 - 06/13 (week 16) : 
- 06/07 - 06/13 (week 17) : oral defense
- 06/07 - 06/13 (week 18) : ???

