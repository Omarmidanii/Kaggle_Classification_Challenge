from pathlib import Path
import json, time, warnings, zipfile, os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

DATA_DIR=Path('/mnt/data/fite')
ART=Path('/mnt/data/level5_artifacts')
ART.mkdir(exist_ok=True)
train=pd.read_csv(DATA_DIR/'train_data.csv').reset_index(drop=True)
test=pd.read_csv(DATA_DIR/'test_data.csv').reset_index(drop=True)
sample=pd.read_csv(DATA_DIR/'sample_submission.csv')
feature_cols=[c for c in train.columns if c.startswith('f')]
cont_features=['f1','f2','f9','f10','f14','f20']
binary_features=[c for c in feature_cols if c not in cont_features]
le=LabelEncoder(); y=le.fit_transform(train['target']); class_names=list(le.classes_)
X=train[feature_cols].copy(); X_test=test[feature_cols].copy()

# ---------------- Feature engineering ----------------
def add_rule_features(df):
    out=df.copy()
    thr={'f10':0.00605,'f14':0.064465,'f20':0.1425,'f1':0.845,'f2':0.016,'f9':0.1535}
    for f,t in thr.items():
        out[f'{f}_gt_thr']=(out[f] > t).astype(int)
        out[f'{f}_dist_thr']=(out[f]-t).abs()
    out['region_class1_like']=((out['f10']>thr['f10']) & (out['f14']<=thr['f14']) & (out['f20']<=thr['f20']) & (out['f1']<=thr['f1'])).astype(int)
    out['region_class2_like']=((out['f10']>thr['f10']) & (out['f14']>thr['f14']) & (out['f12']<=0.5) & (out['f9']<=thr['f9']) & (out['f17']<=0.5)).astype(int)
    out['region_class3_like']=((out['f10']<=thr['f10']) | ((out['f10']>thr['f10']) & (out['f14']>thr['f14']) & (out['f12']>0.5))).astype(int)
    # boundary risk: closer to threshold = higher risk after normalization
    out['min_boundary_distance'] = np.minimum.reduce([
        (out['f10']-thr['f10']).abs(),
        (out['f14']-thr['f14']).abs(),
        (out['f20']-thr['f20']).abs(),
        (out['f2']-thr['f2']).abs(),
        (out['f9']-thr['f9']).abs(),
    ])
    return out
X_eng=add_rule_features(X)
X_test_eng=add_rule_features(X_test)

# ---------------- Folds ----------------
def make_id_block_folds(df, n_splits=5, id_col='ID'):
    sorted_indices=df.sort_values(id_col).index.to_numpy()
    blocks=np.array_split(sorted_indices, n_splits)
    all_indices=df.index.to_numpy()
    return [(np.setdiff1d(all_indices, block), block) for block in blocks]
folds_random=list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
folds_block=make_id_block_folds(train, n_splits=5, id_col='ID')

# ---------------- Adversarial weights ----------------
def adversarial_train_weights():
    all_df=pd.concat([X, X_test], axis=0, ignore_index=True)
    y_adv=np.r_[np.zeros(len(X)), np.ones(len(X_test))]
    adv=RandomForestClassifier(n_estimators=200, max_depth=5, max_features='sqrt', random_state=42, n_jobs=1)
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_adv=cross_val_predict(adv, all_df, y_adv, cv=cv, method='predict_proba', n_jobs=1)[:,1]
    auc=roc_auc_score(y_adv, p_adv)
    p_train=np.clip(p_adv[:len(X)], 0.05, 0.95)
    w=p_train/(1-p_train)
    w=w/np.mean(w)
    w=np.clip(w, 0.5, 2.0)
    pd.DataFrame({'ID':train['ID'],'adv_p_test':p_train,'sample_weight':w}).to_csv(ART/'level5_adversarial_train_weights.csv', index=False)
    return w, auc
adv_weights, adv_auc = adversarial_train_weights()

# ---------------- Models ----------------
models={
    'decision_tree_depth4': (DecisionTreeClassifier(max_depth=4, random_state=42), 'base', None),
    'decision_tree_depth5': (DecisionTreeClassifier(max_depth=5, random_state=42), 'base', None),
    'random_forest_depth8_bal': (RandomForestClassifier(n_estimators=220, max_depth=8, max_features='sqrt', class_weight='balanced_subsample', random_state=42, n_jobs=1), 'base', None),
    'extra_trees_bal': (ExtraTreesClassifier(n_estimators=220, max_features='sqrt', class_weight='balanced', min_samples_leaf=1, random_state=42, n_jobs=1), 'base', None),
    'gradient_boosting': (GradientBoostingClassifier(n_estimators=140, learning_rate=0.04, max_depth=2, random_state=42), 'base', None),
    'lgbm_regularized': (LGBMClassifier(objective='multiclass', n_estimators=120, learning_rate=0.04, num_leaves=7, max_depth=3, min_child_samples=12, reg_lambda=2.5, subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1, n_jobs=1, force_col_wise=True), 'base', None),
    'lgbm_engineered': (LGBMClassifier(objective='multiclass', n_estimators=120, learning_rate=0.04, num_leaves=7, max_depth=3, min_child_samples=12, reg_lambda=2.5, subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1, n_jobs=1, force_col_wise=True), 'eng', None),
    'lgbm_adv_weighted': (LGBMClassifier(objective='multiclass', n_estimators=120, learning_rate=0.04, num_leaves=7, max_depth=3, min_child_samples=12, reg_lambda=2.5, subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1, n_jobs=1, force_col_wise=True), 'base', adv_weights),
    'logreg_engineered_bal': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.25, class_weight='balanced', random_state=42)), 'eng', None),
}

def get_data(which):
    return (X_eng, X_test_eng) if which=='eng' else (X, X_test)

def fit_with_optional_weights(model, Xtr, ytr, weights):
    if weights is None:
        model.fit(Xtr, ytr)
    else:
        # pipelines may need special handling, but weighted models here are not pipelines.
        model.fit(Xtr, ytr, sample_weight=weights)
    return model

def evaluate_model(model, which, folds, name, predict_test=False, sample_weights=None):
    Xuse, Xtuse=get_data(which)
    oof=np.zeros((len(train), len(class_names)))
    testp=np.zeros((len(test), len(class_names))) if predict_test else None
    fold_rows=[]
    t0=time.time()
    for fold,(tr,va) in enumerate(folds, start=1):
        m=clone(model)
        w = sample_weights[tr] if sample_weights is not None else None
        fit_with_optional_weights(m, Xuse.iloc[tr], y[tr], w)
        p=m.predict_proba(Xuse.iloc[va])
        oof[va]=p
        pred=p.argmax(axis=1)
        fold_rows.append({'model':name,'fold':fold,'n_valid':len(va),'accuracy':accuracy_score(y[va],pred),'balanced_accuracy':balanced_accuracy_score(y[va],pred),'macro_f1':f1_score(y[va],pred,average='macro')})
        if predict_test:
            testp += m.predict_proba(Xtuse)/len(folds)
    pred=oof.argmax(axis=1)
    return {
        'model':name,'feature_set':which,'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),
        'log_loss':log_loss(y,oof,labels=list(range(len(class_names)))),'n_errors':int((pred!=y).sum()),'seconds':time.time()-t0,
    }, pd.DataFrame(fold_rows), oof, testp

rows=[]; fold_scores=[]; oofs={}; tests={}; block_oofs={}
for name,(model,which,weights) in models.items():
    print('RUN', name, flush=True)
    row, fs, oof, testp=evaluate_model(model, which, folds_random, name, predict_test=True, sample_weights=weights)
    row['validation']='stratified_5fold'
    rows.append(row); fold_scores.append(fs.assign(validation='stratified_5fold'))
    oofs[name]=oof; tests[name]=testp
    rowb, fsb, oofb, _=evaluate_model(model, which, folds_block, name, predict_test=False, sample_weights=weights)
    rowb['validation']='id_block_5fold'
    rows.append(rowb); fold_scores.append(fsb.assign(validation='id_block_5fold'))
    block_oofs[name]=oofb
    print(name, 'strat', row['accuracy'], 'block', rowb['accuracy'], flush=True)

metrics=pd.DataFrame(rows)
metrics['robust_score']=0.50*metrics['accuracy']+0.25*metrics['balanced_accuracy']+0.25*metrics['macro_f1']
metrics.to_csv(ART/'level5_model_comparison.csv', index=False)
pd.concat(fold_scores, ignore_index=True).to_csv(ART/'level5_fold_scores.csv', index=False)

# ---------------- Ensembles ----------------
selected=['decision_tree_depth5','lgbm_regularized','gradient_boosting','random_forest_depth8_bal','extra_trees_bal','lgbm_engineered','lgbm_adv_weighted']
def weighted_sum(probas, weights):
    active={n:w for n,w in weights.items() if w>0 and n in probas}
    return sum(active[n]*probas[n] for n in active)/sum(active.values())
configs={
    'ensemble_equal_all': {n:1 for n in selected},
    'ensemble_tree_boost': {'decision_tree_depth5':1.5,'lgbm_regularized':1.2,'gradient_boosting':1.2},
    'ensemble_robust_shift': {'decision_tree_depth5':1.0,'lgbm_regularized':1.0,'lgbm_adv_weighted':1.0,'random_forest_depth8_bal':0.7,'gradient_boosting':1.0},
    'ensemble_engineered': {'decision_tree_depth5':1.0,'lgbm_engineered':1.2,'lgbm_regularized':1.0,'gradient_boosting':0.8},
}
ens_rows=[]; ens_oofs={}; ens_tests={}
for cname,w in configs.items():
    p=weighted_sum(oofs,w); pt=weighted_sum(tests,w); pred=p.argmax(axis=1)
    row={'candidate':cname,'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'log_loss':log_loss(y,p,labels=list(range(len(class_names)))),'n_errors':int((pred!=y).sum()),'weights':json.dumps(w)}
    ens_rows.append(row); ens_oofs[cname]=p; ens_tests[cname]=pt

# Random search for weights, constrained to avoid overfitting too much.
rng=np.random.default_rng(123)
top=selected
best=None
for i in range(1200):
    alpha=np.array([1.5 if n in ['decision_tree_depth5','lgbm_regularized','gradient_boosting'] else 1.0 for n in top])
    arr=rng.dirichlet(alpha)
    w=dict(zip(top,arr))
    p=weighted_sum(oofs,w); pred=p.argmax(axis=1)
    acc=accuracy_score(y,pred); bal=balanced_accuracy_score(y,pred); macro=f1_score(y,pred,average='macro'); ll=log_loss(y,p,labels=list(range(len(class_names))))
    objective=acc + 0.05*macro + 0.03*bal - 0.002*ll
    if best is None or objective>best['objective']:
        best={'weights':w,'objective':objective,'accuracy':acc,'balanced_accuracy':bal,'macro_f1':macro,'log_loss':ll,'n_errors':int((pred!=y).sum())}
p=weighted_sum(oofs,best['weights']); pt=weighted_sum(tests,best['weights'])
ens_rows.append({'candidate':'ensemble_weight_search','accuracy':best['accuracy'],'balanced_accuracy':best['balanced_accuracy'],'macro_f1':best['macro_f1'],'log_loss':best['log_loss'],'n_errors':best['n_errors'],'weights':json.dumps({k:float(v) for k,v in best['weights'].items()})})
ens_oofs['ensemble_weight_search']=p; ens_tests['ensemble_weight_search']=pt
ens_df=pd.DataFrame(ens_rows).sort_values(['accuracy','macro_f1'],ascending=False)
ens_df.to_csv(ART/'level5_ensemble_comparison.csv', index=False)

# ---------------- Conservative threshold postprocess ----------------
def conservative_postprocess(proba, rare_threshold):
    pred=proba.argmax(axis=1).copy()
    conf=proba.max(axis=1)
    c3=list(class_names).index('class3')
    for i,c in enumerate(pred):
        if class_names[c] in ['class1','class2'] and conf[i] < rare_threshold:
            pred[i]=c3
    return pred
thr_rows=[]
for cand in list(ens_oofs.keys())+['lgbm_regularized','decision_tree_depth5']:
    p=ens_oofs[cand] if cand in ens_oofs else oofs[cand]
    for thr in [0.50,0.60,0.70,0.80,0.90,0.95,0.98,0.99]:
        pred=conservative_postprocess(p,thr)
        thr_rows.append({'candidate':cand,'rare_threshold':thr,'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'n_errors':int((pred!=y).sum()),'n_class1_pred':int((pred==0).sum()),'n_class2_pred':int((pred==1).sum()),'n_class3_pred':int((pred==2).sum())})
thr_df=pd.DataFrame(thr_rows).sort_values(['accuracy','macro_f1'],ascending=False)
thr_df.to_csv(ART/'level5_conservative_thresholds.csv', index=False)

# ---------------- Submissions ----------------
def save_sub(name, proba=None, pred_int=None):
    if pred_int is None:
        pred_int=proba.argmax(axis=1)
    target=le.inverse_transform(pred_int)
    sub=sample[['ID']].copy(); sub['target']=target
    path=ART/f'submission_level5_{name}.csv'
    sub.to_csv(path,index=False)
    dist=sub['target'].value_counts().rename_axis('target').reset_index(name='count')
    dist['share']=dist['count']/len(sub)
    dist.to_csv(ART/f'distribution_submission_level5_{name}.csv', index=False)
    return path
sub_paths=[]
for name in ['lgbm_regularized','gradient_boosting','random_forest_depth8_bal','lgbm_adv_weighted']:
    sub_paths.append(save_sub(name, proba=tests[name]))
for name in ['ensemble_tree_boost','ensemble_robust_shift','ensemble_engineered','ensemble_weight_search']:
    sub_paths.append(save_sub(name, proba=ens_tests[name]))
for thr in [0.90,0.95,0.98]:
    sub_paths.append(save_sub(f'ensemble_weight_search_conservative_{str(thr).replace(".","")}', pred_int=conservative_postprocess(ens_tests['ensemble_weight_search'],thr)))

# Differences vs Level 4 tree depth5
prev=Path('/mnt/data/level4_artifacts/submission_level4_tree_depth5.csv')
if prev.exists():
    prev_df=pd.read_csv(prev)
    diff_rows=[]
    for path in sub_paths:
        cand=pd.read_csv(path)
        diff_rows.append({'submission':Path(path).name,'different_from_level4_depth5':int((cand['target']!=prev_df['target']).sum())})
    pd.DataFrame(diff_rows).to_csv(ART/'level5_submission_differences_vs_level4_depth5.csv', index=False)

# Best candidate error analysis
best_cand=ens_df.iloc[0]['candidate']
best_oof=ens_oofs[best_cand]
best_pred=best_oof.argmax(axis=1)
err_mask=best_pred!=y
errors=train.loc[err_mask, ['ID','target']+feature_cols].copy()
errors['predicted']=le.inverse_transform(best_pred[err_mask])
errors['confidence']=best_oof[err_mask].max(axis=1)
for i,c in enumerate(class_names): errors[f'proba_{c}']=best_oof[err_mask,i]
errors.to_csv(ART/'level5_best_candidate_validation_errors.csv', index=False)
pd.DataFrame(confusion_matrix(y,best_pred), index=[f'true_{c}' for c in class_names], columns=[f'pred_{c}' for c in class_names]).to_csv(ART/'level5_best_candidate_confusion_matrix.csv')
with open(ART/'level5_best_candidate_classification_report.txt','w') as f: f.write(classification_report(y,best_pred,target_names=class_names,digits=5))

summary={
    'level':'Level 5 - Robust models, engineered rule features, adversarial weighting, ensembles',
    'class_names':class_names,
    'n_train':len(train),'n_test':len(test),'n_base_features':len(feature_cols),'n_engineered_features':X_eng.shape[1],
    'adversarial_validation_auc_for_weights':float(adv_auc),
    'models_tested':list(models.keys()),
    'best_ensemble_by_stratified_cv':best_cand,
    'best_ensemble_metrics':ens_df.iloc[0].to_dict(),
    'submission_files':[Path(p).name for p in sub_paths]
}
with open(ART/'level5_run_summary.json','w') as f: json.dump(summary,f,indent=2)
print('DONE')
print(json.dumps(summary, indent=2))
print('\nModel comparison')
print(metrics.sort_values(['validation','accuracy'], ascending=[True,False]).to_string(index=False))
print('\nEnsembles')
print(ens_df.to_string(index=False))
print('\nThreshold top')
print(thr_df.head(10).to_string(index=False))
