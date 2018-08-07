from autokeras.classifier import load_image_dataset
from autokeras.classifier import ImageClassifier

if __name__ == '__main__':
	x_train, y_train = load_image_dataset(csv_file_path="train2/label.csv",
	                                      images_path="train2")
	print(x_train.shape)
	print(y_train.shape)

	x_test, y_test = load_image_dataset(csv_file_path="test2/label.csv",
	                                    images_path="test2")
	print(x_test.shape)
	print(y_test.shape)


	clf = ImageClassifier()
	clf.fit(x_train, y_train)
	results = clf.predict(x_test)

