#JLGouws - 19G4436
#I think I should get around 70/80
#I did not get my code to work on the kddcup99 dataset
#It will not work for dataset with categorical features
#I tried to keep everything relatively general, and it should work on most
#scikit datasets without categorical features
#I did grid search for all models except for XGB, which had problems with running a grid search
#I don't print out the parameters of the model and preprocessing
#Maybe I should realistically get 65/80, but it depends on how other students did with their task
#At the en of the day the assessment is comparative
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.utils import Bunch
from pathlib import Path
import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.pipeline import Pipeline
import pandas as pd
import sklearn.preprocessing as prep



random_state = 42

#This plots a VFM
def pca_map(X, figsize=(10,10), sup="", print_values= False, output_pdf = None):
    import matplotlib.pyplot as plt
    #PCA
    columns=X.columns.values
    pca=PCA(n_components=2)
    pca.fit(X)
    pca_values=pca.components_
    
    #Plot
    fig = plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': 14}) 
    
    #Plot circle
    x=np.linspace(start=-1,stop=1,num=500)
    y_positive=lambda x: np.sqrt(1-x**2) 
    y_negative=lambda x: -np.sqrt(1-x**2)
    plt.plot(x,list(map(y_positive, x)), color='maroon')
    plt.plot(x,list(map(y_negative, x)),color='maroon')
    
    #Plot smaller circle
    x=np.linspace(start=-0.5,stop=0.5,num=500)
    y_positive=lambda x: np.sqrt(0.5**2-x**2) 
    y_negative=lambda x: -np.sqrt(0.5**2-x**2)
    plt.plot(x,list(map(y_positive, x)), color='maroon')
    plt.plot(x,list(map(y_negative, x)),color='maroon')
    
    #Create broken lines
    x=np.linspace(start=-1,stop=1,num=30)
    plt.scatter(x,[0]*len(x), marker='_',color='maroon')
    plt.scatter([0]*len(x), x, marker='|',color='maroon')

    #Define color list
    colors = ['blue', 'red', 'green', 'black', 'purple', 'brown']
    if len(pca_values[0]) > 6:
        colors=colors*(int(len(pca_values[0])/6)+1)

    #Plot arrow
    add_string=""
    for i in range(len(pca_values[0])):
        xi=pca_values[0][i]
        yi=pca_values[1][i]
        plt.arrow(0,0, 
                  dx=xi, dy=yi, 
                  head_width=0.03, head_length=0.03, 
                  color=colors[i], length_includes_head=True)
        if print_values==True:
            add_string=f" ({round(xi,2)} {round(yi,2)})"
        plt.text(pca_values[0, i], 
                 pca_values[1, i] , 
                 s=columns[i] + add_string )

    plt.xlabel(f"Component 1 ({round(pca.explained_variance_ratio_[0]*100,2)}%)")
    plt.ylabel(f"Component 2 ({round(pca.explained_variance_ratio_[1]*100,2)}%)")
    plt.title('Variable factor map (PCA)')
    plt.suptitle(sup, y=1, fontsize=18)
    if (output_pdf is None) or (output_pdf == ""):
        plt.show()
    else:
        fig.savefig(output_pdf, format = 'pdf')

def preprocessImages(X, images):
    from skimage.feature import hog
    from skimage.color import rgb2gray
    if (images.shape[1] >= 25 and images.shape[2] >= 25):
        data = []
        for image in images:#perform hog on the images if they are big enough
            data += [hog(image, orientations = 9, pixels_per_cell=(9,9), channel_axis = -1 if len(images.shape) - 1 == 3 else None)]
        X = np.array(data)
    elif len(images.shape) - 1 == 3:
        X = rgb2gray(images)
    return X

def preprocessNonImages(X):
    #TODO I am not sure what extra preprocessing to do here, I already do PCA and scaling below
    return X

def preprocess_data(X, images = None):
    global preprocess_data
    X = preprocessImages(X, images) if(images is not None) else preprocessNonImages(X)
    #using PCA for feature selection, keep 99% of variance
    scaler = Pipeline([('PCA', PCA(n_components = 0.99, svd_solver = 'full')), ('scaler', prep.MinMaxScaler())])#min max scaling because it is safe
    scaler.fit(X)#fit the scaler to the test data
    X = scaler.transform(X)
    def new_prep(X, images = None):
        X = preprocessImages(X, images) if(images is not None) else preprocessNonImages(X)
        X = scaler.transform(X)
        return X
    preprocess_data = new_prep #kind of a closure to keep the scaler pipeline
    return X

def k_nearest(X, y):
    from sklearn.neighbors import KNeighborsClassifier as knn

    tune_param =[
        {'n_neighbors': [1, 2, 3, 5, 7, 10, 15, 20]},#not sure what other parameters to score on
    ]

    grid = GridSearchCV(knn(), tune_param, cv=5, scoring='f1_weighted', n_jobs = -1)

    grid.fit(X, y)

    model = grid.best_estimator_ #get the best model predicted by the grid search
    return model


def logistic_regression(X, y):
    from sklearn.linear_model import LogisticRegression as LR

    tune_param =[
            {'C': [1e-2, 1e-1, 1, 5, 10, 20], 'penalty': ['l1', 'l2'], 'solver' : ['saga', 'liblinear']}, #parameter for logistic regression
    ]

    grid = GridSearchCV(LR(random_state = random_state, max_iter = 3000, warm_start = True)
            , tune_param, cv=5, scoring='f1_weighted', n_jobs = -1)

    grid.fit(X, y)

    model = grid.best_estimator_ #get the best model predicted by the grid search
    return model

def random_forest(X, y):
    from sklearn.ensemble import RandomForestClassifier as RFC

    tune_param =[
        {'criterion': ['gini', 'entropy'], 'max_depth': [None, 50, 100, 200], 'n_estimators': [50, 100, 500]},
    ]

    grid = GridSearchCV(RFC(random_state = random_state, warm_start = True, n_jobs = -1), tune_param, cv=5, scoring='f1_weighted', n_jobs = -1)

    grid.fit(X, y) #note how we do CV on the training set

    model = grid.best_estimator_ #get the best model predicted by the grid search
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

    grid = GridSearchCV(SVC(), tune_param, cv=5, scoring='f1_weighted', n_jobs = -1)

    grid.fit(X, y) #note how we do CV on the training set

    model = grid.best_estimator_ #get the best model predicted by the grid search

    return model


def mlp(X, y):
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(solver='lbfgs', warm_start=True, max_iter = 5000, random_state=random_state)

    n_features = X.shape[-1]

    #tune the model
    tune_param = [
        {'hidden_layer_sizes': [(2 * n_features//3 + 1, 4 * n_features//9 + 1), 
            (n_features//2 + 1, n_features//4 + 1), (2 * n_features//3 + 1), 
            (n_features//2 + 1), (n_features//3 + 1), (n_features//4 + 1), ()], 
            'alpha': [1e-1, 1, 1.5, 3], 'activation': ['relu', 'logistic', 'identity']
        }
    ]

    grid = GridSearchCV(mlp, tune_param, cv=5, scoring='f1_weighted', n_jobs = -1)

    grid.fit(X, y)

    model = grid.best_estimator_ #get the best model predicted by the grid search

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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'micro')
    return precision, accuracy, recall, f1


def compare_models(models, X_test, y_test, args):
    import matplotlib.pyplot as plt
    import seaborn as sns
    model_dict = {'Model' : args.classifiers.split(','), 'precision' : [], 'accuracy' : [], 'recall' : [], 'f1' : []}
    for model in models:
        #get scores on test sets
        precision, accuracy, recall, f1 = score_metrics(model, X_test, y_test)
        model_dict['precision'] += [precision]
        model_dict['accuracy'] += [accuracy]
        model_dict['recall'] += [recall]
        model_dict['f1'] += [f1]
    fig, axes = plt.subplots(3, 1, figsize = (len(model_dict['Model']) * 5, 21))
    results_df = pd.DataFrame(model_dict)
    #Plot the precision, accuracy and recall of the models
    sns.barplot(x="Model", y="precision", data = results_df, ax = axes[0])
    sns.barplot(x="Model", y="accuracy", data = results_df, ax = axes[1])
    sns.barplot(x="Model", y="recall", data = results_df, ax = axes[2])
    plt.savefig("ModelsCompared.pdf")
    #get the list to compare models
    try:
        compareList = model_dict[args.decisionmetric]
    except Exception as e:
        print('You entered', args.decisionmetric, 'which is an invalid metric.')
        compareList = model_dict['precision']
    print('The best classifier was determined to be: ', model_dict['Model'][compareList.index(max(compareList))])
    return

def visualize_images(df, output_pdf, images):
    import seaborn as sns
    import matplotlib.pyplot as plt
    #just plot sample images from each class, don't really know what else to plot here
    fig, ax = plt.subplots(1, np.unique(df['Class']).size)
    for i, e in enumerate(np.dstack(np.unique(df['Class'], return_index = True))[0]):
        ax[i].imshow(images[e[1]], cmap = 'gray')
        ax[i].spines[['left', 'bottom', 'top', 'right']].set_visible(False);
        ax[i].set(xticks = [], yticks = [], title = e[0])
    fig.savefig(output_pdf, format = 'pdf')
    return

def visualize_non_images(df, output_pdf):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    df_scaled = df
    df_scaled[df.columns[:-1]] = MinMaxScaler().fit_transform(df[df.columns[:-1]])
    DfMelted = pd.melt(df_scaled, id_vars="Class", 
                       var_name = "features", value_name = 'value')
    num_plots = df.columns[:-1].size // 5 + 1
    for i, features in enumerate(np.array_split(df.columns[:-1], num_plots)):
        #apparently this works
        fig = sns.pairplot(data = df_scaled, hue = 'Class', vars = features).fig
        fig.savefig(output_pdf, format = 'pdf')

    num_plots = df.columns[:-1].size // 10 + 1
    for i, features in enumerate(np.array_split(df.columns[:-1], num_plots)):
        fig = plt.figure(figsize=(30,12))
        sns.stripplot(x="features", y="value", hue="Class", jitter = 0.2,
                data=DfMelted[DfMelted["features"].isin(features)], alpha = 0.8) #

        plt.xticks(rotation=30)
        fig.savefig(output_pdf, format = 'pdf')

    pca_map(df.drop('Class', axis = 1), output_pdf = output_pdf)
    return

def visualize_data(df, images = None):
    import matplotlib.pyplot as plt
    from matplotlib.backends import backend_pdf
    import seaborn as sns
    output_pdf = backend_pdf.PdfPages('vizualization.pdf')
    if(images is not None): visualize_images(df, output_pdf, images)
    else: visualize_non_images(df, output_pdf)

    #distribution of class labels
    fig = plt.figure(figsize=(2.5 * np.unique(df['Class']).size, 6))
    sns.histplot(x="Class", hue="Class", shrink=0.75, data=df)

    fig.savefig(output_pdf, format = 'pdf')

    output_pdf.close()
    return

def encode_kddcup99(dataset):
    #tried to hot encode this,but it doesn't work
    from sklearn.feature_extraction.text import CountVectorizer
    X = prep.OneHotEncoder(handle_unknown='ignore').fit_transform(dataset.data)
    return X

# Entry Point of Program
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='../images', help = 'The dataset that you which to work with e.g. breast_cancer') #I changed this for my ease of
                                                               #access
    p.add_argument('--classifiers', type=str, help = 'The classifiers that you wish to use these should be given as a comma separated list e.g. mlp,k_nearest,svm')
    p.add_argument('--decisionmetric', type=str, default='precision', help = 'The metric to choose the best classifier')
    args = p.parse_args()
    print("Welcome to the multiple model classifier. I see that you have chosen", args.classifiers, "as your model(s) of choice.")

    # Load Dataset
    if args.dataset == '../images':
        dataset = load_image_files(args.dataset)
    else:
        try:
            from sklearn import datasets
            dataset = getattr(datasets, 'load_' + args.dataset)()#should work with most sklearn datasets
        except Exception as e:
            try:
                dataset = getattr(datasets, 'fetch_' + args.dataset)()
            except Exception as e1:
                print(e, e1, sep='\n')

    X = dataset.data
    y = dataset.target

    if args.dataset == 'kddcup99':
        X = encode_kddcup99(dataset)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=random_state)

    train_df = pd.DataFrame(X_train, columns = dataset.feature_names)
    train_df['Class'] = dataset.target_names[y_train]

    train_images, test_images = None, None
    if hasattr(dataset, 'images'):
        train_images, test_images = train_test_split(dataset.images, y, test_size=0.3, random_state=random_state)[0:2]  
    # Visualize Data

    visualize_data(train_df, train_images)

    # Preprocess Data
    X_train_new = preprocess_data(X_train, train_images)

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
    X_test_new = preprocess_data(X_test, test_images)

    # If multiple models selected: make a bar graph comparing them. If not, just report on results
    if len(selected_models) > 1:
        compare_models(selected_models, X_test_new, y_test, args)
    else:
        prec, acc, recall, f1 = score_metrics(selected_models[0], X_test_new, y_test)
        print("Accuracy for model", args.classifiers ,":", acc)
        print("Precision for model", args.classifiers ,":", prec)
        print("Recall for model", args.classifiers ,":", recall)
