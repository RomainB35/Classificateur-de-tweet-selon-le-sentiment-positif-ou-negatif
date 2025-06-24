import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_score, recall_score, f1_score)
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier

def train_optimized_classifier(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: list,
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    grid_search: bool = True,
    verbose: bool = True,
    sample_size: int = None
) -> dict:
    """
    Entraîne un classifieur optimisé avec gestion des targets 0/4 et échantillonnage aléatoire.
    
    Args:
        df: DataFrame contenant les données
        target_column: Nom de la colonne cible (doit contenir 0 et 4)
        numeric_features: Liste des colonnes numériques à utiliser
        model_type: Type de modèle ('random_forest', 'xgboost', 'logistic_regression' ou 'stacking')
        test_size: Proportion pour le test set (0-1)
        random_state: Seed pour reproductibilité
        cv_folds: Nombre de folds pour la validation croisée
        grid_search: Si True, effectue une recherche d'hyperparamètres
        verbose: Si True, affiche les détails de l'entraînement
        sample_size: Taille de l'échantillon aléatoire (None pour tout le dataset)
    
    Returns:
        Dictionnaire contenant le modèle entraîné, les métriques et les importances
    """
    
    # Configuration des warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # 1. Échantillonnage aléatoire si spécifié
        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), 
                          random_state=random_state)
            if verbose:
                print(f"\nÉchantillon aléatoire de {len(df)} instances sélectionné")
        
        # 2. Vérification des valeurs de la target
        unique_targets = sorted(df[target_column].unique())
        if unique_targets != [0, 4]:
            raise ValueError(f"La target doit contenir uniquement 0 et 4. Valeurs trouvées: {unique_targets}")
        
        # 3. Préparation des données
        X = df[numeric_features]
        y = df[target_column]
        
        # Conversion des labels pour XGBoost et Logistic Regression (0 reste 0, 4 devient 1)
        y_transformed = y.copy()
        if model_type in ['xgboost', 'logistic_regression', 'stacking']:
            y_transformed = y.replace({4: 1})
            class_names = ['0', '4']
            if verbose:
                print(f"Conversion des labels: 4 -> 1 pour {model_type}")
        else:
            class_names = ['0', '4']
        
        # Split train-test stratifié
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, 
            test_size=test_size, 
            stratify=y_transformed,
            random_state=random_state
        )
        
        # 4. Configuration des modèles
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=random_state)
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 5, 10],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }
            
        elif model_type == 'xgboost':
            base_model = XGBClassifier(
                random_state=random_state,
                eval_metric='logloss'
            )
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 6],
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0]
            }
            
        elif model_type == 'logistic_regression':
            base_model = LogisticRegression(random_state=random_state, max_iter=1000)
            # Paramètres mieux organisés pour éviter les combinaisons incompatibles
            param_grid = [
                {
                    'classifier__penalty': ['l1'],
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__solver': ['liblinear', 'saga']
                },
                {
                    'classifier__penalty': ['l2'],
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
                },
                {
                    'classifier__penalty': ['elasticnet'],
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__l1_ratio': [0.1, 0.5, 0.9],
                    'classifier__solver': ['saga']
                },
                {
                    'classifier__penalty': [None],
                    'classifier__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
                }
            ]
            
        elif model_type == 'stacking':
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
                ('xgb', XGBClassifier(eval_metric='logloss', random_state=random_state)),
                ('lr', LogisticRegression(max_iter=1000, random_state=random_state))
            ]
            
            meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
            
            base_model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=cv_folds,
                n_jobs=-1
            )
            
            param_grid = {
                'classifier__rf__n_estimators': [100, 200],
                'classifier__xgb__learning_rate': [0.01, 0.1],
                'classifier__lr__C': [0.1, 1, 10],
                'classifier__final_estimator__C': [0.1, 1, 10]
            }
        
        # 5. Création du pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', base_model)
        ])
        
        # 6. Grid Search avec Cross-Validation
        if grid_search:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1 if verbose else 0,
                error_score='raise'  # Pour debugger les erreurs
            )
            
            try:
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                
                if verbose:
                    print("\nMeilleurs paramètres trouvés:")
                    print(tabulate(grid.best_params_.items(), headers=['Paramètre', 'Valeur'], tablefmt='grid'))
                    print(f"\nMeilleur score CV: {grid.best_score_:.4f}")
            except Exception as e:
                if verbose:
                    print(f"\nErreur lors de l'entraînement: {str(e)}")
                # Fallback: entraînement sans grid search
                pipeline.fit(X_train, y_train)
                best_model = pipeline
                if verbose:
                    print("Utilisation des paramètres par défaut suite à l'échec de GridSearch")
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
        
        # 7. Prédiction et conversion des labels
        y_pred = best_model.predict(X_test)
        
        # Conversion inverse pour l'évaluation (1 -> 4)
        y_test_eval = y_test.replace({1: 4}) if model_type in ['xgboost', 'logistic_regression', 'stacking'] else y_test
        y_pred_eval = y_pred.copy()
        if model_type in ['xgboost', 'logistic_regression', 'stacking']:
            y_pred_eval[y_pred == 1] = 4
        
        # 8. Calcul des métriques
        metrics = {
            'accuracy': accuracy_score(y_test_eval, y_pred_eval),
            'precision': precision_score(y_test_eval, y_pred_eval, average='weighted'),
            'recall': recall_score(y_test_eval, y_pred_eval, average='weighted'),
            'f1_score': f1_score(y_test_eval, y_pred_eval, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test_eval, y_pred_eval),
            'classification_report': classification_report(
                y_test_eval, 
                y_pred_eval, 
                target_names=class_names, 
                output_dict=True
            ),
            'best_params': grid.best_params_ if grid_search and 'grid' in locals() else None
        }
        
        # 9. Feature Importance
        if model_type == 'random_forest':
            importances = best_model.named_steps['classifier'].feature_importances_
        elif model_type == 'xgboost':
            importances = best_model.named_steps['classifier'].feature_importances_
        elif model_type == 'logistic_regression':
            # Pour la régression logistique, on prend les valeurs absolues des coefficients
            if hasattr(best_model.named_steps['classifier'], 'coef_'):
                importances = np.abs(best_model.named_steps['classifier'].coef_[0])
            else:
                importances = np.zeros(len(numeric_features))
        elif model_type == 'stacking':
            rf_imp = best_model.named_steps['classifier'].estimators_[0].feature_importances_
            xgb_imp = best_model.named_steps['classifier'].estimators_[1].feature_importances_
            lr_imp = np.abs(best_model.named_steps['classifier'].estimators_[2].coef_[0]) if hasattr(
                best_model.named_steps['classifier'].estimators_[2], 'coef_') else np.zeros(len(numeric_features))
            # Normalisation des importances pour les combiner
            rf_imp = rf_imp / rf_imp.sum() if rf_imp.sum() > 0 else rf_imp
            xgb_imp = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp
            lr_imp = lr_imp / lr_imp.sum() if lr_imp.sum() > 0 else lr_imp
            importances = (rf_imp + xgb_imp + lr_imp) / 3
        
        feature_importances = pd.Series(
            importances,
            index=numeric_features
        ).sort_values(ascending=False)
        
        # 10. Affichage des résultats
        if verbose:
            # Affichage des métriques
            print("\n" + "="*60)
            print(f"{' '*20}RÉSULTATS FINAUX ({model_type.upper()})")
            print("="*60)
            
            print("\nMétriques principales:")
            print(tabulate([
                ["Accuracy", f"{metrics['accuracy']:.4f}"],
                ["Precision", f"{metrics['precision']:.4f}"],
                ["Recall", f"{metrics['recall']:.4f}"],
                ["F1-Score", f"{metrics['f1_score']:.4f}"]
            ], headers=["Métrique", "Valeur"], tablefmt="grid"))
            
            # Matrice de confusion
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                metrics['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title('Matrice de Confusion', pad=20)
            plt.ylabel('Vraies étiquettes')
            plt.xlabel('Étiquettes prédites')
            plt.show()
            
            # Feature Importance
            plt.figure(figsize=(10, 6))
            feature_importances.plot(kind='bar')
            plt.title('Importance des Variables')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Rapport de classification
            print("\nRapport de classification détaillé:")
            print(tabulate(
                pd.DataFrame(metrics['classification_report']).transpose().round(4),
                headers='keys',
                tablefmt='grid'
            ))
        
        # 11. Retour des résultats
        return {
            'model': best_model,
            'metrics': metrics,
            'feature_importances': feature_importances,
            'class_names': class_names,
            'cv_results': grid.cv_results_ if grid_search and 'grid' in locals() else None,
            'model_type': model_type,
            'sample_size': len(df)
        }


import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Télécharger le modèle VADER si nécessaire
nltk.download('vader_lexicon', quiet=True)

def add_sentiment_word_counts_vader(df, text_column):
    sia = SentimentIntensityAnalyzer()

    def extract_sentiment_counts(text):
        scores = sia.polarity_scores(str(text))
        # Il n’y a pas de "nombre de mots" positifs/négatifs exacts, mais on peut approximer :
        # - pos score = proportion de sentiment positif
        # - neg score = proportion de sentiment négatif
        # On retourne des scores entre 0 et 1, on peut les multiplier par nb de mots si tu veux un count approx.
        return scores['pos'], scores['neg']

    sentiment_scores = df[text_column].astype(str).apply(extract_sentiment_counts)
    df['vader_pos_score'] = sentiment_scores.apply(lambda x: x[0])
    df['vader_neg_score'] = sentiment_scores.apply(lambda x: x[1])
    return df
