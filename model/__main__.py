from sklearn.pipeline import make_pipeline
from model.data import DataManager, ImageResizer, RleEncoder, ResultSaver, ShaperReporter
from model.cnn import CnnClassifier


from sklearn.pipeline import make_pipeline

def build_model():
    classifier = make_pipeline(
        ShaperReporter(),
        CnnClassifier()
    )
    return classifier


def main():
    data = DataManager()
    train_imges_, train_masks_ = data.images(), data.masks()

    train_imges = ImageResizer().transform(train_imges_)
    train_masks = ImageResizer().transform(train_masks_)

    classifier = build_model()

    print("\nTraining...")
    classifier.fit(train_imges, train_masks)


    print("\nPredicting...")
    predicted = classifier.predict(train_imges)
    print(predicted.shape)

    predictions = data.imagelist[["ImageId"]]
    predictions['predicted'] = list(predicted)

    output = RleEncoder().transform(predictions)
    print(output)

    ResultSaver().transform(output)

    # Fixes the issue described here
    # https://github.com/tensorflow/tensorflow/issues/8652
    del classifier


if __name__ == '__main__':
    main()