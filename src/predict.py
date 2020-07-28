from model import Model
import argparse
import os

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Testing on models generated from train.py on the EMNIST dataset')
    parser.add_argument('--model', type=str, default='./models/model.h5', help='Directory path containing the model')
    parser.add_argument('--data', type=str, default="./data/test/test.jpg", help='path of testing file')
    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.data):
        print("Invalid parameters provided")
        exit()

    test_model = Model()
    test_model.predict(model_path=args.model, img_path=args.data)
