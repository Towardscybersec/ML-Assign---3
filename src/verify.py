"""Audit required evidence, provenance, leakage controls and saved metric consistency."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from common import *

def main():
    checks={}; failures=[]
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition: failures.append(name)
    p=ROOT/'artifacts/splits'; r=ROOT/'artifacts/results'; t=r/'test'
    z=np.load(p/'split.npz'); check('one_split_correct_sizes',len(z['train'])==45000 and len(z['val'])==5000)
    check('split_disjoint_unique',len(np.unique(np.r_[z['train'],z['val']]))==50000)
    check('seed_from_roll',int(''.join(filter(str.isdigit,CFG['identity']['roll'])))==260012)
    frozen=json.loads((r/'frozen.json').read_text())
    for file,sha in frozen['sha256'].items(): check('frozen:'+file,(ROOT/file).exists() and digest(ROOT/file)==sha)
    check('test_matches_freeze',json.loads((t/'complete.json').read_text())['freeze_sha256']==digest(r/'frozen.json'))
    table=pd.read_csv(t/'classifier_table.csv'); check('all_ten_classifier_comparisons',len(table)==10)
    for name in table.run:
        h=pd.read_json(ROOT/f'artifacts/runs/{name}/history.json'); ck=load_ckpt(ROOT/f'artifacts/runs/{name}/best.pt')
        check('selection:'+name,abs(float(h.val_loss.min())-ck['val_loss'])<1e-8)
        check('budget:'+name,len(h)==CFG['classifier']['epochs'])
        check('finite_history:'+name,np.isfinite(h[['train_loss','val_loss','train_accuracy','val_accuracy']].to_numpy()).all())
        recomputed=metrics(np.load(t/f'{name}_logits.npy'),np.load(t/'labels.npy'))
        saved=table[table.run==name].iloc[0]
        check('classifier_metrics:'+name,all(abs(recomputed[key]-float(saved[key]))<1e-10 for key in ['accuracy','nll','brier','ece']))
    a=pd.read_csv(t/'autoencoders.csv'); check('three_clean_aes_and_dae',set(a.representation)=={'ae_16','ae_64','ae_256','dae'})
    chosen=json.loads((r/'ae_selection.json').read_text()); clean=a[a.representation!='dae']
    check('ae_validation_selection',int(clean.sort_values(['val_mse','dimension']).dimension.iloc[0])==chosen['dim'])
    sweep=pd.read_csv(r/'svm_sweep.csv'); check('18_svm_settings',len(sweep)==18)
    check('6_C_5_decades',sweep.C.nunique()==6 and sweep.C.max()/sweep.C.min()>=1000)
    check('three_kernels_including_rbf',set(sweep.kernel)=={'linear','rbf','poly'})
    check('svm_converged',(sweep.fit_status==0).all())
    selection=json.loads((r/'svm_selection.json').read_text()); check('svm_validation_selection',abs(selection['val_accuracy']-sweep.val_accuracy.max())<1e-12)
    import joblib
    check('representation_scaler_train_only',joblib.load(r/'representation_scaler.joblib').n_samples_seen_==45000)
    probe_sweep=pd.read_csv(r/'probe_sweep.csv')
    check('all_16_probe_settings',len(probe_sweep)==16)
    check('probes_converged',(probe_sweep.iterations<CFG['probe']['max_iter']).all())
    for name in ['ae_16','ae_64','ae_256','dae']:
        probe=joblib.load(r/f'probes/{name}_selected.joblib')
        check('probe_scaler_train_only:'+name,probe['scaler'].n_samples_seen_==45000)
    pairs=pd.read_csv(t/'pair_examples.csv'); check('three_pairs_with_both_sources',pairs.pair.nunique()>=3 and len(pairs)==36 and set(pairs.source)=={'test','support'})
    check('pair_specific_support_nonzero',(pairs[pairs.source=='support'].pair_dual_coefficient.abs()>1e-12).all())
    svm_pred=np.load(t/'svm_predictions.npy'); y=np.load(t/'labels.npy'); sm=json.loads((t/'svm.json').read_text())
    check('svm_accuracy_from_predictions',abs(float((svm_pred==y).mean())-sm['accuracy'])<1e-12)
    check('svm_once',sm['evaluation_calls']==1)
    from itertools import combinations
    scores=np.load(t/'svm_ovo_scores.npy'); votes=np.zeros((len(y),10),dtype=int)
    for col,(i,jj) in enumerate(combinations(range(10),2)):
        votes[:,i]+=(scores[:,col]>0); votes[:,jj]+=(scores[:,col]<=0)
    check('actual_svm_ovo_sign_matches_votes',np.array_equal(votes.argmax(1),svm_pred))
    # Independently reconstruct pair scores from the kernel and active coefficients.
    bundle=joblib.load(r/'svm/selected.joblib'); model=bundle['model']
    xx=bundle['scaler'].transform(np.load(t/'cnn_features.npy')[:7]).astype(np.float64)
    sv=model.support_vectors_; dot=xx@sv.T
    if model.kernel=='linear': kernel=dot
    elif model.kernel=='rbf':
        d2=np.maximum(0,(xx*xx).sum(1)[:,None]+(sv*sv).sum(1)[None,:]-2*dot)
        kernel=np.exp(-model._gamma*d2)
    elif model.kernel=='poly': kernel=(model._gamma*dot+model.coef0)**model.degree
    svclasses=np.repeat(model.classes_,model.n_support_)
    for col,(i,jj) in enumerate(combinations(range(10),2)):
        coef=np.where(svclasses==i,model.dual_coef_[jj-1],np.where(svclasses==jj,model.dual_coef_[i],0.))
        manual=kernel@coef+model.intercept_[col]
        check(f'pair_kernel_reconstruction:{i}-{jj}',np.allclose(manual,scores[:7,col],atol=1e-6,rtol=1e-5))
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
    assignments=np.load(t/'cluster_assignments_before_labels.npy'); cm=json.loads((t/'clustering.json').read_text())
    check('cluster_ari_recomputed',abs(adjusted_rand_score(y,assignments)-cm['ARI'])<1e-12)
    check('cluster_nmi_recomputed',abs(normalized_mutual_info_score(y,assignments)-cm['NMI'])<1e-12)
    check('clusters_assigned_before_labels',not json.loads((t/'cluster_assignment_provenance.json').read_text())['test_labels_accessed'])
    contingency=pd.read_csv(t/'cluster_contingency.csv',index_col=0).to_numpy()
    check('contingency_covers_test',contingency.sum()==10000 and np.all(contingency.sum(0)==1000))
    for name in ['classifier_learning','memorisation','confusions','reliability','ae_learning','ae_reconstructions','denoising_reconstructions','reconstruction_vs_semantics','svm_sweep','pair_0','pair_1','pair_2','cluster_contingency']:
        check('figure:'+name,(ROOT/f'figures/{name}.pdf').exists())
    mem=pd.read_json(ROOT/'artifacts/runs/memorisation/history.json'); check('memorisation_fits_corruption',mem.corrupt_target_accuracy.iloc[-1]>=.8)
    check('memorisation_full_budget',len(mem)==CFG['memorisation']['epochs'])
    report=ROOT/'report/ML_Assignment_3_Vaibhav_Krishna.pdf'
    if report.exists():
        from pypdf import PdfReader
        pdf=PdfReader(report); first=pdf.pages[0].extract_text(); alltext='\n'.join(a.extract_text() for a in pdf.pages)
        reference_start=next((i for i,p in enumerate(pdf.pages) if any(line.strip()=='References' for line in p.extract_text().splitlines())),len(pdf.pages))
        check('report_body_page_limit',reference_start<=10)
        # Kerning can make PDF extractors insert spaces inside names.
        compact=''.join(first.split())
        check('report_name_roll_first_page','VaibhavKrishna' in compact and 'P26CS0012' in compact)
        check('report_no_placeholder','TODO' not in alltext and 'INSERT' not in alltext)
        source_commit=(ROOT/'report/source_commit.txt').read_text().strip()
        check('source_commit_on_page1',source_commit in first.replace('\n','').replace(' ',''))
        for file,sha in frozen['sha256'].items():
            blob=subprocess.run(['git','show',source_commit+':'+file],cwd=ROOT,capture_output=True)
            import hashlib
            check('committed_frozen:'+file,blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==sha)
    save_json(r/'verification.json', {'checks': checks, 'passed': sum(checks.values()), 'total': len(checks), 'failures': failures})
    print(json.dumps({'passed':sum(checks.values()),'total':len(checks),'failures':failures},indent=2))
    if failures: sys.exit(1)
if __name__=='__main__': main()
