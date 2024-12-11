import sys
from dataset import Dataset
from preprocessing import Eigenspectrum, Preprocessor
#from adv_statistics import negentropy, differences_entropy(A)
#from adv_statistics import negentropy, differences_entropy

if __name__ == '__main__':
	#warnings.filterwarnings('ignore')
@@ -24,12 +24,14 @@
	parser.add_option("-l", "--load_data", action="store_true", dest="load_data")
	parser.add_option("-e", "--extract_features", action="store_true", dest="extract_features")
	parser.add_option("-s", "--speaker_indipendence", action="store_true", dest="speaker_indipendence")
	parser.add_option("-i", "--plot_eigenspectrum", action="store_true", dest="plot_eigenspectrum")
	(options, args) = parser.parse_args(sys.argv)
	load_data = options.load_data
	extract_features = options.extract_features
	db_type = options.db_type
	speaker_indipendence = options.speaker_indipendence
	path = options.path
	plot_eigenspectrum = options.plot_eigenspectrum

	if load_data:
		print "Loading data from " + db_type + " dataset..."
@@ -48,8 +50,8 @@
	print "Number of dataset samples: " + str(n_samples)

	if extract_features:
		win_size = 0.05
		step = 0.025
		win_size = 0.04
		step = 0.01
		Fglobal = []
		i = 0
		for (x,Fs) in db.data:
@@ -69,10 +71,6 @@
	Fglobal = np.array(Fglobal)
	y = np.array(db.targets)

	# eigenspectrum over all data
	es = Eigenspectrum(Fglobal)
	es.show()
	# evaluating SVM using cross validation
	print "Evaluating model with cross validation..."

@@ -85,16 +83,25 @@
		splits = sss.split(Fglobal, y)

	# setting preprocessing
	pp = Preprocessor('standard',n_components=45)
	pp = Preprocessor('standard',n_components=50)
	n_classes = len(db.classes)
	clf = OneVsRestClassifier(svm.SVC(kernel='rbf',C=10, gamma=0.01))
	prfs = []; scores = []; acc = np.zeros(n_classes)
	mi_threshold = 0.0
	for (train,test) in splits:
		# selecting features using mutual information
		Ftrain = Fglobal[train]; Ftest = Fglobal[test]
		f_subset = pp.mutual_info_select(Ftrain,y[train],0.0)
		f_subset = pp.mutual_info_select(Ftrain,y[train],mi_threshold)
		Ftrain = Ftrain[:,f_subset]; Ftest = Ftest[:,f_subset]
		
		#standard transformation
		(Ftrain,Ftest) = pp.standardize(Ftrain,Ftest)
		
		# eigenspectrum over all data
		if plot_eigenspectrum:
			es = Eigenspectrum(Ftrain)
			es.show()
		
		(Ftrain,Ftest) = pp.project_on_pc(Ftrain,Ftest)

		clf.fit(Ftrain, y[train])
