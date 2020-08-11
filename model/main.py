import click
from cnn import CnnClassifier
from data import (DataReader, ResultSaver, RleEncoder,
                  ShaperReporter, TrainDataReader)

from metrics import iou
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from timer import Timer


def build_model():
    classifier = make_pipeline(
        ShaperReporter(),
        CnnClassifier()
    )
    return classifier


@click.command()
def main():
    pass


@main.command()
@click.option("--path", type=click.Path(exists=True), default="data/cells")
def train(path):
    with Timer('Prepare data'):
        train_imges, train_masks = TrainDataReader(sample=10).read()
        X_train, X_test, y_train, y_test = train_test_split(
            train_imges, train_masks, test_size=0.3, random_state=42)

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
        test_ids, test_images = DataReader(sample=10).read()

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
