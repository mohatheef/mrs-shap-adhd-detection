from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from catboost import CatBoostClassifier

def get_base_models():
    """Return base machine learning models with hyperparameters matching the manuscript."""
    # Support Vector Machine (SVM)
    svm = SVC(
        C=10.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    
    # Logistic Regression (LR)
    lr = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    # Gradient Boosting (GB)
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    
    # CatBoost Classifier
    cat = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=5,
        auto_class_weights='Balanced',
        verbose=0,
        random_state=42
    )
    
    return {
        'svm': svm,
        'lr': lr,
        'gb': gb,
        'cat': cat
    }

def build_voting_ensemble(models=None):
    """Build the soft Voting Ensemble model using SVM, LR, GB, and CatBoost."""
    if models is None:
        models = get_base_models()
        
    ensemble = VotingClassifier(
        estimators=[
            ('cat', models['cat']),
            ('svm', models['svm']),
            ('gb', models['gb']),
            ('lr', models['lr'])
        ],
        voting='soft',
        # CatBoost (0.3), SVM (0.3), GB (0.25), LR (0.15)
        weights=[0.3, 0.3, 0.25, 0.15]
    )
    return ensemble
