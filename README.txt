This is a skin detection application and I have used 2 different approaches for this. All the codes are thoroughly commented to understand each and every step.
1) Skin detection using face detection:
	skinDetectThresh.py uses face detection to eventually detect skin/non-skin pixels. Three main steps are:
	a) Detect face using dlib face detector & extract a patch for reference (nose or cheek patch)
	b) Extract all pixels similar to the reference (using cv2.inrange() function) and generate mask
	c) Apply mask to remove non-skin pixels from input image

	How to use:
	python skinDetectThresh.py
	(Uses default image of Hillary clinton)
	python skinDetectThresh.py -f "path-to-image"
	(Uses specified image)

	Advantages:
	No training time cost. Hence faster than most of the Machine Learning algorithms

	Drawbacks:
	Depends on face detection. If no face is detected in image, thresholds cannot be specified.

	Used cases:
	1) images/hillary_clinton.jpg : works almost perfect.
	2) images/humanhand1.jpg : No face detected, hence fails.
	3) images/img2.jpg : works almost perfect.
	4) images/running.jpg : No face detected, hence fails.
	5) images/sideFace.jpg : works almost perfect. (Face detector performs 						exceptionally well here to detect face in the image)
	6) images/tennis.jpg : No face detected, hence fails.

	Results:
	Please checkout the results in SkinDetection/results/Threshold/


2) Skin detection using Naive Bayes Classifier:
	skinDetectML.py uses Naive Bayes Classifier trained on dataset available at "http://cs-chan.com/downloads_skin_dataset.html". Classifier predicts whether a given pixel is a skin pixel/non-skin pixel.
	Three main steps are:
	1) Downloading training data and preprocessing it to make it ready to train a classifier
	2) Trainin a Naive Bayes Classifier using training data
	3) Testing an image to see the results

	How to use:
	python skinDetectML.py
	(Uses default image of Hillary clinton and default colorspace YCbCr)
	python skinDetectThresh.py -f "path-to-image" -c "colorspace(y OR b)"
	(Uses specified image and colorspace to produce skin detection results)

	Advantages:
	Does not depend on face detection, hence works well as a generalized skin detector and not facial skin detector. Can perform even better, if dataset is expanded to include more images.
	
	Drawbacks:
	Training time is the cost that we have to pay for the more generalized algorithm. Also requires more memory in order to store trained model.

	Used cases:
	1) images/hillary_clinton.jpg : works almost perfect.
	2) images/humanhand1.jpg : works almost perfect, unlike method (1).
	3) images/img2.jpg : works almost perfect.
	4) images/running.jpg : works almost perfect, unlike method(1).
	5) images/sideFace.jpg : works almost perfect.
	6) images/tennis.jpg : works almost perfect, unlike method(1).

	Results:
	Please checkout the results in SkinDetection/results/NaiveBayesClassifier/



