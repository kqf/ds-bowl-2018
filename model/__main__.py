from model.data import DataManager
from model.cnn import CnnClassifier

def main():
    # get train_data
    data = DataManager()
    train_imges, train_masks = data.images(), data.masks()

    # get u_net model
    classifier = CnnClassifier()

    print("\nTraining...")
    print(train_imges.shape)
    print(train_masks.shape)
    classifier.fit(train_imges, train_masks)

if __name__ == '__main__':
    main()