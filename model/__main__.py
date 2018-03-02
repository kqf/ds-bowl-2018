from model.data import DataManager, ImageResizer, RleEncoder, ResultSaver, ShaperReporter
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
        data = DataManager()
        train_imges_, train_masks_ = data.images(), data.masks()

        train_imges = ImageResizer().transform(train_imges_)
        train_masks = ImageResizer().transform(train_masks_)

    with Timer('Prepare data'):
        classifier = build_model()


    with Timer('Training the model'):
        classifier.fit(train_imges, train_masks)


    with Timer("Reading test set"):
        test_images_ = data.test()
        test_images = ImageResizer().transform(test_images_)

    with Timer('Predicting the values'):
        predicted = classifier.predict(test_images)
        print(predicted.shape)


    with Timer('Save the results'):
        predictions = test_images_.imagelist[['ImageId']]
        predictions['predicted'] = list(predicted)

        output = RleEncoder().transform(predictions)
        ResultSaver().transform(output)

    # Fixes the issue described here
    # https://github.com/tensorflow/tensorflow/issues/8652
    del classifier


if __name__ == '__main__':
    main()