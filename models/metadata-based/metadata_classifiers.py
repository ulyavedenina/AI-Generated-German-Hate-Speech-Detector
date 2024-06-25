import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import time

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

df = pd.read_csv('../../../dataset/bot_human_metadata.tsv', sep='\t', encoding='utf-8', engine='python')

y = df['human/bot']
X = df.drop(['human/bot'], axis=1)


# num humans
len([element for element in y if element == 0])
# num bots
len([element for element in y if element == 1])

from scipy.stats import zscore

column_types = X.dtypes

columns_to_convert = ['is_gold', 'is_mod']
X[columns_to_convert] = X[columns_to_convert].astype(int)
X['has_verified_email'] = X['has_verified_email'].map(lambda x: 1 if x else 0)

column_types = X.dtypes

print(X.columns)

X_scaled = X.apply(zscore)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.head())

y_train = y_train.values
y_test= y_test.values

X_train = X_train.values
X_test = X_test.values

# Define models with their best hyperparameters
models = {
    'DecisionTreeClassifier': {'model': DecisionTreeClassifier(random_state= 42, criterion='log_loss', max_depth=5, max_features='log2', splitter='random'), 'params': {}},
    'KNeighborsClassifier': {'model': KNeighborsClassifier(algorithm='auto', n_neighbors=7, weights='uniform'), 'params': {}},
    'SVC': {'model': SVC(random_state= 42, C=0.025, gamma='auto', kernel='linear', probability=True), 'params': {}},
    'NuSVC': {'model': NuSVC(random_state= 42, gamma='auto', kernel='linear', probability=True), 'params': {}},
    'RandomForestClassifier': {'model': RandomForestClassifier(random_state= 42, criterion='entropy', max_features='log2', min_samples_split=8, n_estimators=10), 'params': {}},
    'AdaBoostClassifier': {'model': AdaBoostClassifier(random_state= 42, learning_rate=0.01, n_estimators=50), 'params': {}},
    'GradientBoostingClassifier': {'model': GradientBoostingClassifier(random_state= 42, learning_rate=0.01, n_estimators=10), 'params': {}},
    'GaussianNB': {'model': GaussianNB(), 'params': {}},
    'LinearDiscriminantAnalysis': {'model': LinearDiscriminantAnalysis(solver='svd'), 'params': {}},
    'QuadraticDiscriminantAnalysis': {'model': QuadraticDiscriminantAnalysis(), 'params': {}}
}

# Run evaluations on the test set for each model
for model_name, model_data in models.items():
    print("=" * 30)
    print(model_name)
    start_time = time.time()
    
    # Initialize and train the model
    model = model_data['model']
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    test_predictions = model.predict(X_test)
    test_predictions_prob = model.predict_proba(X_test)
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_test, test_predictions)
    fscore = f1_score(y_test, test_predictions)
    ll = log_loss(y_test, test_predictions_prob)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print('****Results****')
    print("Time:", elapsed_time, "seconds")
    print("Accuracy: {:.4%}".format(acc))
    print("F1-score: {:.4%}".format(fscore))
    print("Log Loss:", ll)
    print("=" * 30)
