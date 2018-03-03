from model.data import TrainDataReader, ImageResizer, RleEncoder, ResultSaver, ShaperReporter, DataReader
from model.cnn import CnnClassifier
from model.timer import Timer

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

    with Timer('Prepare data'):
        classifier = build_model()

    with Timer('Training the model'):
        classifier.fit(train_imges, train_masks)

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