import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from pathlib import Path
import skimage
from skimage.io import imread
from skimage.transform import resize


random_state = 42

def preprocess_data(X):
    return X


def k_nearest(X, y):
    from sklearn.neighbors import KNeighborsClassifier as knn
    model = knn(n_neighbors = 3)
    model.fit(X, y)
    return model


def logistic_regression(X, y):
    from sklearn.linear_model import LosgisticRegression as LR
    model = LR(random_state = random_state)
    model.fit(X, y)
    return model

def random_forest(X, y):
    from sklearn.ensemble import RandomForestClassifer as RFC
    model = RFC(random_state = random_state)
    model.fit(X, y)
    return model


def xgboost(X,y):
    try:
        from xgboost import XGBClassifier as XGB
        model = XGB(use_label_encoder = False, eval_metric='logloss', random_state = random_state)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as XGB
        model = XGB(random_state = random_state)
    model.fit(X, y)
    return model

def svm(X, y):
    # TODO: Initialize SVM, and train
    quit()


def mlp(X, y):
    # TODO: Initialize MLP, and train
    quit()


model_map = {
    'k_nearest': k_nearest,
    'logistic_regression': logistic_regression,
    'random_forest': random_forest,
    'xgboost': xgboost,
    'svm': svm,
    'mlp': mlp
}


def load_image_files(container_path, dimension=(30, 30)):

    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Your own dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    # return in the exact same format as the built-in datasets
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


def score_metrics(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    return precision, accuracy, recall


def compare_models(models, X_test, y_test):
    # TODO: Draw Bar Graph(s) showing accuracy, precision and recall
    # TODO: report on best model for precision, accuracy and recall
    return

def visualize_data(X, y):
    # TODO: visualize the data
    return

# Entry Point of Program
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='../images') #I changed this for my ease of
                                                               #access
    p.add_argument('--classifiers', type=str)
    args = p.parse_args()
    print("Welcome to the multiple model classifier. I see that you have chosen", args.classifiers, "as your model(s) of choice.")

    # Load Dataset
    if args.dataset == '../images':
        dataset = load_image_files(args.dataset)
    else:
        try:
            from sklearn import datasets
            dataset = getattr(datasets, 'load_' + args.dataset)()
        except Exception as e:
            print(e)

    images = hasattr(dataset, 'images')

    X = dataset.data
    y = dataset.target

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=random_state)

    # Visualize Data
    visualize_data(X_train, y_train)

    # Preprocess Data
    X_train_new = preprocess_data(X_train)

    # Splits the command-line arguments into separate classifiers
    classifiers = args.classifiers.split(',')

    # Iterate over selected classifiers and create a model based on the choice of classifier
    selected_models = []
    for cl in classifiers:
        model = model_map[cl]
        trained_model = model(X_train_new, y_train)

        # Append a trained model to selected_models
        selected_models.append(trained_model)

    # Preprocess Test data
    X_test_new = preprocess_data(X_test)

    # If multiple models selected: make a bar graph comparing them. If not, just report on results
    if len(selected_models) > 1:
        compare_models(selected_models, X_test_new, y_test)
    else:
        prec, acc, recall = score_metrics(selected_models[0], X_test_new, y_test)
        print("Accuracy for model", args.classifiers ,":", acc)
        print("Precision for model", args.classifiers ,":", prec)
        print("Recall for model", args.classifiers ,":", recall)
