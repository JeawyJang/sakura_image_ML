from autokeras.classifier import load_image_dataset

x_train, y_train = load_image_dataset(csv_file_path="train/label.csv",
                                      images_path="train")
print(x_train.shape)
print(y_train.shape)

x_test, y_test = load_image_dataset(csv_file_path="test/label.csv",
                                    images_path="test")
print(x_test.shape)
print(y_test.shape)
