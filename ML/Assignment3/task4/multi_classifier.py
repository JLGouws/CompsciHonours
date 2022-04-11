import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import Bunch
from pathlib import Path
import skimage
from skimage.io import imread
from skimage.transform import resize
import pandas as pd


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
    from sklearn.svm import SVC

    tune_param =[
                    {'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [1e-1, 1e-2, 1e-3], 'C': [0.01, 1, 10]},
    ]

    grid = GridSearchCV(SVC(), tune_param, cv=5, scoring='f1_weighted')

    grid.fit(X, y) #note how we do CV on the training set

    model = grid.best_estimator_ #get the best model predicted by the grid search

    return model


def mlp(X, y):
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(solver='lbfgs', warm_start=True, max_iter = 5000, random_state=random_state)

    tune_param = [
        {'hidden_layer_sizes': [(20, 5), (20, 3), (20), (10), (5), (4), (3)], 'alpha': [1e-1, 1, 1.5, 3], 'activation': ['relu', 'logistic', 'identity']},
    ]

    grid= GridSearchCV(mlp, tune_param, cv=5, scoring='f1_weighted')

    grid.fit(X, y) #note how we do CV on the training set

    model = grid.best_estimator_ #get the best model predicted by the grid search
    print("shape of x", X.shape)
    print("model inputs", model.n_features_in_)
    print("model outputs", model.n_outputs_)

    return model

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
    categories = np.array(categories)

    # return in the exact same format as the built-in datasets
    return Bunch(data=flat_data,
                 target=target,
                 feature_names=[f'pixel{i}' for i in range(3 * dimension[0] * dimension[1])],
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
    for model in models:
        print(score_metrics(model, X_test, y_test))
    return

def visualize_images(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    return

def visualize_non_images(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    df_scaled = df
    df_scaled[df.columns[:-1]] = MinMaxScaler().fit_transform(df[df.columns[:-1]])
    DfMelted = pd.melt(df_scaled, id_vars="Class", 
                       var_name = "features", value_name = 'value')
    num_plots = df.columns[:-1].size // 5 + 1
    for i, features in enumerate(np.array_split(df.columns[:-1], num_plots)):
        sns.pairplot(data = df_scaled, hue = 'Class', vars = features)

    num_plots = df.columns[:-1].size // 10 + 1
    for i, features in enumerate(np.array_split(df.columns[:-1], num_plots)):
        plt.figure(figsize=(30,12))
        sns.stripplot(x="features", y="value", hue="Class", jitter = 0.2,
                data=DfMelted[DfMelted["features"].isin(features)], alpha = 0.8) #

        plt.xticks(rotation=30)

    plt.show()
    return

def visualize_data(df, are_images = False):
    if(are_images): visualize_images(df)
    else: visualize_non_images(df)
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

    are_images = hasattr(dataset, 'images')

    X = dataset.data
    y = dataset.target

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=random_state)

    print(y_train)
    print(y_train.shape)
    train_df = pd.DataFrame(X_train, columns = dataset.feature_names)
    train_df['Class'] = dataset.target_names[y_train]

    # Visualize Data
    visualize_data(train_df, are_images)

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
