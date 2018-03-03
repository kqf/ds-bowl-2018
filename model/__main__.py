from model.data import TrainDataReader, ImageResizer, RleEncoder, ResultSaver, ShaperReporter, DataReader
from model.cnn import CnnClassifier
from model.timer import Timer
from metrics import iou

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
def build_model():
    classifier = make_pipeline(
        ShaperReporter(),
        CnnClassifier()
    )
    return classifier


def main():
    with Timer('Prepare data'):
        train_imges, train_masks = TrainDataReader().read()
        X_train, X_test, y_train, y_test = train_test_split(train_imges, train_masks, test_size=0.3, random_state=42)

    with Timer('Prepare data'):
        classifier = build_model()

    with Timer('Training the model'):
        classifier.fit(X_train, X_train)

    with Timer('Evaluate performance'):
        train_predict = classifier.predict(X_train)
        test_predict = classifier.predict(X_test)
        print()
        print("-------- Model evaluation -----------")
        print("Score on training set:", iou(y_train, train_predict))

        print("Score on test set:", iou(y_test, test_predict))
        print("-------- ---------------- -----------")


    with Timer("Reading test set"):
        test_ids, test_images = DataReader().read()

    with Timer('Predicting the values'):
        predicted = classifier.predict(test_images)

    with Timer('Save the results'):
        test_ids['predicted'] = list(predicted)
        output = RleEncoder().transform(test_ids)
        ResultSaver().transform(output)

    # Fixes the issue described here
    # https://github.com/tensorflow/tensorflow/issues/8652
    del classifier


if __name__ == '__main__':
    main()