# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

models = [
    {
        'clf_name': 'Dummy',
        'clf': DummyClassifier(random_state=1),
        'has_scaling': True,
        'search_params': {},
        'best_params': {},
        'best_score': 0  # weighted F1-Score
    },
    {
        'clf_name': 'Logistic Regression',
        'clf': LogisticRegression(class_weight='balanced', multi_class='multinomial', n_jobs=-1, random_state=1),
        'has_scaling': True,
        'search_params': {
            'clf__penalty': ['l1', 'l2', 'elasticnet'],
            'clf__tol': [1, 0.1, 0.01, 0.001, 0.0001, 1e-05],
            'clf__C': [0.001, 0.01, 0.1, 1, 10],
            'clf__solver': ['sag', 'saga', 'newton-cg']
        },
        'best_params': {
            'clf__tol': 1e-05,
            'clf__solver': 'saga',
            'clf__penalty': 'l2',
            'clf__C': 0.001
        },
        'best_score': 0.8391989484388706
    },
    {
        'clf_name': 'Random Forest',
        'clf': RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=1),
        'has_scaling': False,
        'search_params': {
            'clf__bootstrap': [True, False],
            'clf__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'clf__max_features': ['auto', 'sqrt'],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__min_samples_split': [2, 5, 10],
            'clf__n_estimators': [100, 200, 300, 400, 500]
        },
        'best_params': {
            'clf__n_estimators': 500,
            'clf__min_samples_split': 5,
            'clf__min_samples_leaf': 1,
            'clf__max_features': 'auto',
            'clf__max_depth': 30,
            'clf__bootstrap': False
        },
        'best_score': 0.9939005405682252
    },
    {
        'clf_name': 'Decision Tree',
        'clf': DecisionTreeClassifier(class_weight='balanced', random_state=1),
        'has_scaling': False,
        'search_params': {
            'clf__criterion': ['gini', 'entropy', 'log_loss'],
            'clf__splitter': ['best', 'random'],
            'clf__max_features': ['auto', 'sqrt', 'log2'],
            'clf__max_depth': list(range(3, 18, 3)),
            'clf__min_samples_leaf': [3, 5, 10, 15, 20],
            'clf__min_samples_split': list(range(8, 22, 2)),
        },
        'best_params': {
            'clf__splitter': 'best',
            'clf__min_samples_split': 14,
            'clf__min_samples_leaf': 3,
            'clf__max_features': 'auto',
            'clf__max_depth': 15,
            'clf__criterion': 'log_loss'
        },
        'best_score': 0.9763104573565381,
    },
    {
        'clf_name': 'Naive Bayes',
        'clf': ComplementNB(),
        'has_scaling': False,
        'search_params': {
            'clf__alpha': [0.01, 0.1, 1, 10],
        },
        'best_params': {
            'clf__alpha': 0.01
        },
        'best_score': 0.6254139068432872,
    },
    {
        'clf_name': 'KNN',
        'clf': KNeighborsClassifier(n_jobs=-1),
        'has_scaling': True,
        'search_params': {
            'clf__n_neighbors': list(range(5, 60, 10)),
            'clf__weights': ['uniform', 'distance'],
            'clf__algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'clf__leaf_size': list(range(10, 60, 10)),
            'clf__p': [1, 2]
        },
        'best_params': {
            'clf__weights': 'distance',
            'clf__p': 2,
            'clf__n_neighbors': 5,
            'clf__leaf_size': 50,
            'clf__algorithm': 'ball_tree'
        },
        'best_score': 0.9604647579792767,
    },
    {
        'clf_name': 'SVM',
        'clf': SVC(class_weight='balanced', random_state=1),
        'has_scaling': True,
        'search_params': {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': [0.001, 0.01, 0.1, 10]
        },
        'best_params': {
            'clf__kernel': 'rbf',
            'clf__gamma': 10,
            'clf__C': 10
        },
        'best_score': 0.9549206834280316
    }
]
