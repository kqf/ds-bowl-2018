from model.data import DataManager, ImageResizer, RleEncoder, ResultSaver
from model.cnn import CnnClassifier


from sklearn.pipeline import make_pipeline

def main():
    data = DataManager()
    train_imges_, train_masks_ = data.images(), data.masks()

    train_imges = ImageResizer().transform(train_imges_)
    train_masks = ImageResizer(channels=3).transform(train_masks_)

    print(train_imges.shape)
    print(train_masks.shape)

    classifier = CnnClassifier()
    print("\nTraining...")
    print(train_imges.shape)
    print(train_masks.shape)
    classifier.fit(train_imges, train_masks)


    predictions = data.imagelist[["ImageId"]]
    predicted = classifier.predict(train_imges)
    print(predicted.shape)

    predictions['predicted'] = list(predicted)

    output = RleEncoder().transform(predictions)
    print(output)

    ResultSaver().transform(output)

    # Fixes the issue described here
    # https://github.com/tensorflow/tensorflow/issues/8652
    del classifier


if __name__ == '__main__':
    main()