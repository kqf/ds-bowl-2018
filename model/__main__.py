from model.data import DataManager, ImageResizer, RleEncoder
from model.cnn import CnnClassifier


from sklearn.pipeline import make_pipeline

def main():
    # get train_data
    data = DataManager()
    train_imges_, train_masks_ = data.images(), data.masks()

    train_imges = ImageResizer().transform(train_imges_)
    train_masks = ImageResizer(channels=3).transform(train_masks_)

    print(train_imges.shape)
    print(train_masks.shape)




    # get u_net model
    classifier = CnnClassifier()

    print("\nTraining...")
    print(train_imges.shape)
    print(train_masks.shape)
    classifier.fit(train_imges, train_masks)


    output = data.imagelist[["ImageId"]]
    output['predicted'] = classifier.predict(output)
    output['encoded'] = RleEncoder().transform(
        output
    )


if __name__ == '__main__':
    main()