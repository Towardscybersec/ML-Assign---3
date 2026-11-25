"""The only entry point reading official test data, after a verified selection freeze.
Post-test pair/error inspection is allowed; no model fitting occurs in this file.
"""
import itertools, json, time
import joblib
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from common import *
from selection import get_model, ae_features

@torch.no_grad()
def main():
    seed_all(59)
    out=ROOT/'artifacts/results/test'; out.mkdir(exist_ok=True)
    freeze=ROOT/'artifacts/results/frozen.json'; frozen=json.loads(freeze.read_text())
    for file,sha in frozen['sha256'].items(): assert digest(ROOT/file)==sha, f'Changed after freeze: {file}'
    if (out/'complete.json').exists():
        print('Test evaluation already completed; using saved results, no retesting.',flush=True); return
    from datetime import datetime, timezone
    save_json(out/'started.json',{'frozen_manifest_sha256':digest(freeze),'started_at_utc':datetime.now(timezone.utc).isoformat()})
    ds=CIFAR10(ROOT/'data',train=False,download=False)
    # Test labels intentionally not accessed until the saved cluster assignment below.
    raw=torch.from_numpy(ds.data).permute(0,3,1,2).to(DEVICE); img=Images()
    classifier_logits={}
    for path in sorted((ROOT/'artifacts/runs').glob('*/complete.json')):
        name=path.parent.name
        if not name.startswith(('cnn_','mlp_')): continue
        m,meta=get_model(name); logits=predictions(m,raw,img,meta['condition']); classifier_logits[name]=logits
        np.save(out/f'{name}_logits.npy',logits)
        print('Test predictions',name,flush=True)
    cnn,_=get_model('cnn_clean_17'); cnn_features=predictions(cnn,raw,img,features=True)
    scaler=joblib.load(ROOT/'artifacts/results/representation_scaler.joblib'); fixed=scaler.transform(cnn_features)
    np.save(out/'cnn_features.npy',cnn_features)
    cl=joblib.load(ROOT/'artifacts/results/clustering/selected.joblib'); assigned=cl.predict(fixed)
    np.save(out/'cluster_assignments_before_labels.npy',assigned)
    save_json(out/'cluster_assignment_provenance.json',{'assigned_at_utc':datetime.now(timezone.utc).isoformat(),
        'test_labels_accessed':False,'assignment_sha256':digest(out/'cluster_assignments_before_labels.npy'),
        'model_sha256':digest(ROOT/'artifacts/results/clustering/selected.joblib')})
    svm=joblib.load(ROOT/'artifacts/results/svm/selected.joblib'); sm=svm['model']
    svm_pred=sm.predict(fixed); pair_scores=sm.decision_function(fixed)
    np.save(out/'svm_predictions.npy',svm_pred); np.save(out/'svm_ovo_scores.npy',pair_scores)
    # All unsupervised assignments and selected-model predictions now exist.
    y=np.array(ds.targets); np.save(out/'labels.npy',y)
    cls={name:metrics(logits,y) for name,logits in classifier_logits.items()}
    save_json(out/'classifiers.json',cls)
    rows=[]
    for name,m in cls.items():
        ck=load_ckpt(ROOT/f'artifacts/runs/{name}/best.pt'); meta=ck['meta']
        rows.append({'run':name,'model':meta['kind'],'condition':meta['condition'],'seed':meta['seed'],
             'parameters':meta['parameters'],'selected_epoch':ck['epoch'],**{k:m[k] for k in ['accuracy','nll','brier','ece','incorrect_confidence']}})
    pd.DataFrame(rows).to_csv(out/'classifier_table.csv',index=False)
    clean=img.pixels(raw); noisy=torch.cat([corrupt(a,590002+s) for s,a in zip(range(0,len(clean),256),clean.split(256))])
    # A single tensor supplies identical corrupted inputs to all reconstruction conditions.
    base_logits=np.concatenate([cnn(img.normalise(a)).cpu().numpy() for a in noisy.split(256)])
    denoise={'corrupted':{**metrics(base_logits,y),'mse':float(((noisy-clean)**2).mean().item())}}
    np.save(out/'corrupted_logits.npy',base_logits)
    selected=json.loads((ROOT/'artifacts/results/ae_selection.json').read_text())['dim']
    examples=np.array([np.flatnonzero(y==i)[0] for i in range(10)])
    example_arrays={'indices':examples,'labels':y[examples],'clean':clean[examples].cpu().numpy(),'corrupted':noisy[examples].cpu().numpy()}
    ae_rows=[]
    for name in [f'ae_{d}' for d in CFG['autoencoder']['bottlenecks']]+['dae']:
        m,meta=get_model(name); f=ae_features(m,raw,img); np.save(out/f'{name}_features.npy',f)
        mse_sum=0.
        for start in range(0,len(raw),256): mse_sum+=((m(clean[start:start+256])-clean[start:start+256])**2).sum().item()
        mse=mse_sum/(len(raw)*3072)
        probe=joblib.load(ROOT/f'artifacts/results/probes/{name}_selected.joblib')
        pred=probe['model'].predict(probe['scaler'].transform(f)); np.save(out/f'{name}_probe_predictions.npy',pred)
        ck=load_ckpt(ROOT/f'artifacts/runs/{name}/best.pt')
        ae_rows.append({'representation':name,'dimension':meta['dim'],'parameters':meta['parameters'],
            'val_mse':ck['val_loss'],'clean_test_mse':mse,'clean_test_psnr_from_mean_mse':10*np.log10(1/mse),
            'probe_C':probe['C'],'probe_test_accuracy':float((pred==y).mean()),'epoch':ck['epoch'],
            'val_mse_input':'noisy' if name=='dae' else 'clean'})
        example_arrays[name+'_clean_input']=m(clean[examples]).cpu().numpy()
        if name in [f'ae_{selected}','dae']:
            outputs=[]; mse_sum=0.
            for start in range(0,len(raw),256):
                rec=m(noisy[start:start+256]); outputs.append(cnn(img.normalise(rec)).cpu().numpy()); mse_sum+=((rec-clean[start:start+256])**2).sum().item()
            logits=np.concatenate(outputs); np.save(out/f'{name}_reconstructed_logits.npy',logits)
            denoise[name]={**metrics(logits,y),'mse':mse_sum/(len(raw)*3072)}
            example_arrays[name+'_noisy_input']=m(noisy[examples]).cpu().numpy()
    for m in denoise.values(): m['psnr_from_mean_mse']=float(10*np.log10(1/m['mse']))
    a=base_logits.argmax(1)==y; b=np.load(out/'dae_reconstructed_logits.npy').argmax(1)==y
    denoise['paired_changes']={'noise_wrong_dae_correct':int((~a&b).sum()),'noise_correct_dae_wrong':int((a&~b).sum()),'test_noise_seed_rule':'590002+batch_start; batch_size256; sigma0.2 clip[0,1]'}
    save_json(out/'denoising.json',denoise); pd.DataFrame(ae_rows).to_csv(out/'autoencoders.csv',index=False)
    np.savez_compressed(out/'reconstruction_examples.npz',**example_arrays)
    # Mandatory post-test analysis; no fitting or configuration changes.
    cm=confusion_matrix(y,svm_pred,labels=np.arange(10))
    contingency=np.zeros((cl.n_clusters,10),dtype=int); np.add.at(contingency,(assigned,y),1)
    pd.DataFrame(contingency,columns=ds.classes).to_csv(out/'cluster_contingency.csv',index_label='cluster')
    save_json(out/'clustering.json',{'ARI':adjusted_rand_score(y,assigned),'NMI':normalized_mutual_info_score(y,assigned),'n_clusters':cl.n_clusters,
                 'nmi_average_method':'arithmetic','train_samples':45000,'assignment_before_labels':True})
    candidates=sorted([(int(cm[i,j]+cm[j,i]),i,j) for i,j in itertools.combinations(range(10),2)],key=lambda a:(-a[0],a[1],a[2]))
    chosen=candidates[:3]; pairlist=list(itertools.combinations(range(10),2))
    train_ds=CIFAR10(ROOT/'data',train=True,download=False); subset=svm['subset_global_indices']; train_labels=np.array(train_ds.targets)[subset]
    dev=np.load(ROOT/'artifacts/features/cnn_clean_17.npz')['train']; split=np.load(ROOT/'artifacts/splits/split.npz')['train']; pos={int(a):i for i,a in enumerate(split)}
    svpos=sm.support_; svglobal=subset[svpos]; svlabels=train_labels[svpos]
    svfeatures=scaler.transform(dev[[pos[int(a)] for a in svglobal]])
    np.testing.assert_allclose(svfeatures,sm.support_vectors_,atol=1e-7)
    svscores=sm.decision_function(svfeatures); svpred=sm.predict(svfeatures)
    pairrows=[]; imagerows=[]; imagearrays=[]
    for count,i,j in chosen:
        col=pairlist.index((i,j)); eligible=np.flatnonzero(np.isin(y,[i,j])); near=eligible[np.argsort(np.abs(pair_scores[eligible,col]),kind='stable')[:6]]
        # Only vectors active in THIS binary classifier, not merely in the global SV union.
        pair_dual=np.where(svlabels==i,sm.dual_coef_[j-1],np.where(svlabels==j,sm.dual_coef_[i],0.))
        possible=np.flatnonzero(np.isin(svlabels,[i,j]) & (np.abs(pair_dual)>1e-12))
        assert len(possible)>=6
        near_sv=possible[np.argsort(np.abs(svscores[possible,col]),kind='stable')[:6]]
        # For two random members of these true classes, probability of same cluster.
        a_counts=contingency[:,i]; b_counts=contingency[:,j]
        coassignment=float(np.dot(a_counts,b_counts)/(a_counts.sum()*b_counts.sum()))
        recall_i=cm[i,i]/cm[i].sum(); recall_j=cm[j,j]/cm[j].sum()
        pairrows.append({'class_i':i,'class_j':j,'name_i':ds.classes[i],'name_j':ds.classes[j],
              'i_to_j':int(cm[i,j]),'j_to_i':int(cm[j,i]),'pair_errors':count,'pair_confusion_rate':count/int(np.isin(y,[i,j]).sum()),
              'svm_recall_i':float(recall_i),'svm_recall_j':float(recall_j),'cluster_cross_class_coassignment':coassignment,
              'cluster_modal_i':int(a_counts.argmax()),'cluster_modal_j':int(b_counts.argmax()),
              'cluster_modal_fraction_i':float(a_counts.max()/a_counts.sum()),'cluster_modal_fraction_j':float(b_counts.max()/b_counts.sum())})
        for source,indices,scores,labels,preds,global_ids in [
             ('support',near_sv,svscores,svlabels,svpred,svglobal),
             ('test',near,pair_scores,y,svm_pred,np.arange(len(y)))]:
            for ix in indices:
                global_id=int(global_ids[ix]); score=float(scores[ix,col]); true=int(labels[ix]); pred=int(preds[ix])
                imagerows.append({'pair':f'{i}-{j}','source':source,'global_index':global_id,'true':true,'pred':pred,'ovo_functional_score':score,
                    'positive_favours':i,'negative_favours':j,'row_in_image_npz':len(imagearrays),'pair_dual_coefficient':float(pair_dual[ix]) if source=='support' else None})
                imagearrays.append(train_ds.data[global_id] if source=='support' else ds.data[global_id])
    pd.DataFrame(pairrows).to_csv(out/'difficult_pairs.csv',index=False)
    pd.DataFrame(imagerows).to_csv(out/'pair_examples.csv',index=False)
    np.savez_compressed(out/'pair_images.npz',images=np.array(imagearrays))
    save_json(out/'svm.json',{'accuracy':float((svm_pred==y).mean()),'confusion':cm.tolist(),'n_test':len(y),'evaluation_calls':1,
       'score_type':'libsvm multiclass OVO functional decision; positive favours first class in lexicographic pair; NOT normalised distance',
       'selected_model_sha256':digest(ROOT/'artifacts/results/svm/selected.joblib')})
    save_json(out/'complete.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'freeze_sha256':digest(freeze),'n_test':len(y),'no_selection_after_test':True})
    print('All frozen test measurements and required post-test analyses saved.',flush=True)
if __name__=='__main__': main()
