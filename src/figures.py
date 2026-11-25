"""Build every plotted number directly from saved measurements; no model fitting."""
import json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,CFG,save_json

F=ROOT/'figures'; T=ROOT/'artifacts/results/test'; R=ROOT/'artifacts/results'
NAMES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'figure.dpi':120,'savefig.dpi':200})
def finish(fig,name):
    fig.savefig(F/f'{name}.pdf',bbox_inches='tight'); fig.savefig(F/f'{name}.png',bbox_inches='tight'); plt.close(fig)
def history(name): return pd.read_json(ROOT/f'artifacts/runs/{name}/history.json')
def onset(values,t):
    values=np.array(values)>t
    for i in range(len(values)-1):
        if values[i] and values[i+1]: return i+1
    return None

def main():
    F.mkdir(exist_ok=True)
    table=pd.read_csv(T/'classifier_table.csv'); rows=[]
    for (condition,model),g in table.groupby(['condition','model']):
        rows.append({'condition':condition,'model':model,'parameters':int(g.parameters.iloc[0]),'n_seeds':len(g),
                     'test_mean':float(g.accuracy.mean()),'test_min':float(g.accuracy.min()),'test_max':float(g.accuracy.max()),
                     'sample_sd':float(g.accuracy.std()) if len(g)>1 else None})
    pd.DataFrame(rows).to_csv(R/'classifier_summary.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(7,4.3),constrained_layout=True)
    for col,condition in enumerate(['clean','permuted']):
        for kind,color in [('mlp','tab:orange'),('cnn','tab:blue')]:
            h=history(f'{kind}_{condition}_17')
            axs[0,col].plot(h.epoch,h.train_accuracy,color=color,label=f'{kind.upper()} train')
            axs[0,col].plot(h.epoch,h.val_accuracy,'--',color=color,label=f'{kind.upper()} val')
            axs[1,col].plot(h.epoch,h.train_loss,color=color)
            axs[1,col].plot(h.epoch,h.val_loss,'--',color=color)
        axs[0,col].set(title=condition,ylabel='Accuracy',ylim=(0,1)); axs[0,col].legend(ncol=2)
        axs[1,col].set(xlabel='Epoch',ylabel='Cross-entropy')
    finish(fig,'classifier_learning')
    fig,axs=plt.subplots(1,3,figsize=(7,2.45),constrained_layout=True)
    h=history('memorisation')
    for key,label in [('val_accuracy','Clean val'),('uncorrupted_accuracy','Uncorrupted train'),('corrupt_target_accuracy','Corrupt targets'),('corrupt_original_accuracy','Corrupt originals')]:
        axs[0].plot(h.epoch,h[key],label=label)
        early=h[h.epoch<=20]; axs[1].plot(early.epoch,early[key])
    axs[0].set(title='Full trajectory',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.02)); axs[0].legend(fontsize=7)
    axs[1].set(title='Early learning',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1.02),xlim=(1,20))
    axs[1].axhline(.2,color='gray',lw=.7,ls=':')
    axs[2].plot(h.epoch,h.train_loss,label='Noisy train CE'); axs[2].plot(h.epoch,h.val_loss,label='Clean validation CE')
    axs[2].set(xlabel='Epoch',ylabel='Cross-entropy'); axs[2].legend(fontsize=7)
    finish(fig,'memorisation')
    mem={'useful_epoch':onset(h.val_accuracy,.2),'memorisation_onset_epoch':onset(h.corrupt_target_accuracy,.2),
         'saturation_epoch':onset(h.corrupt_target_accuracy,.8),'last':h.iloc[-1].to_dict(),
         'criterion':'first of two consecutive epochs above prospectively fixed thresholds'}
    save_json(R/'memorisation_analysis.json',mem)
    cls=json.loads((T/'classifiers.json').read_text())
    fig,axs=plt.subplots(2,3,figsize=(7,4.4),constrained_layout=True)
    for row,kind in enumerate(['mlp','cnn']):
        for col,condition in enumerate(['clean','permuted','grayscale']):
            ax=axs[row,col]; cm=np.array(cls[f'{kind}_{condition}_17']['confusion'])/1000
            im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
            ax.set(title=f'{kind.upper()}, {condition}',xticks=range(10),yticks=range(10),xlabel='Predicted',ylabel='True')
            ax.tick_params(labelsize=7)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.65,label='Fraction within true class'); finish(fig,'confusions')
    fig,axs=plt.subplots(1,3,figsize=(7,2.25),constrained_layout=True)
    for ax,condition in zip(axs,['clean','permuted','grayscale']):
        for kind in ['mlp','cnn']:
            b=cls[f'{kind}_{condition}_17']['reliability']; b=[a for a in b if a['n']>0]
            ax.plot([a['confidence'] for a in b],[a['accuracy'] for a in b],marker='.',label=kind.upper())
        ax.plot([0,1],[0,1],'k:',lw=.8); ax.set(title=condition,xlabel='Mean confidence',ylabel='Accuracy',xlim=(0,1),ylim=(0,1)); ax.legend()
    finish(fig,'reliability')
    fig,ax=plt.subplots(figsize=(3.35,2.25),constrained_layout=True)
    for d in [16,64,256]:
        h=history(f'ae_{d}'); ax.plot(h.epoch,h.val_loss,label=f'd={d}')
    ax.set(xlabel='Epoch',ylabel='Validation pixel MSE'); ax.legend(); finish(fig,'ae_learning')
    examples=np.load(T/'reconstruction_examples.npz'); selected=json.loads((R/'ae_selection.json').read_text())['dim']
    def reconstruction_grid(keys,labels,name):
        fig,axs=plt.subplots(len(keys),10,figsize=(7,len(keys)*.70),constrained_layout=True)
        for row,(key,label) in enumerate(zip(keys,labels)):
            for col in range(10):
                ax=axs[row,col]; ax.imshow(examples[key][col].transpose(1,2,0)); ax.set(xticks=[],yticks=[])
                if row==0: ax.set_title(NAMES[col][:5],fontsize=7)
                if col==0: ax.set_ylabel(label,fontsize=7)
                for spine in ax.spines.values(): spine.set_visible(False)
        finish(fig,name)
    reconstruction_grid(['clean','ae_16_clean_input','ae_64_clean_input','ae_256_clean_input'],['Clean','d=16','d=64','d=256'],'ae_reconstructions')
    reconstruction_grid(['clean','corrupted',f'ae_{selected}_noisy_input','dae_noisy_input'],['Clean','Noisy','Clean AE','DAE'],'denoising_reconstructions')
    ae=pd.read_csv(T/'autoencoders.csv'); fig,ax=plt.subplots(figsize=(3.35,2.25),constrained_layout=True)
    for _,row in ae.iterrows(): ax.scatter(row.clean_test_mse,row.probe_test_accuracy); ax.annotate(row.representation,(row.clean_test_mse,row.probe_test_accuracy),xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.set(xlabel='Clean-input test MSE (lower is better)',ylabel='Probe test accuracy'); finish(fig,'reconstruction_vs_semantics')
    sweep=pd.read_csv(R/'svm_sweep.csv'); fig,axs=plt.subplots(2,3,figsize=(7,3.5),constrained_layout=True)
    for col,kernel in enumerate(CFG['svm']['kernels']):
        g=sweep[sweep.kernel==kernel].sort_values('C'); ax=axs[0,col]
        ax.semilogx(g.C,g.train_accuracy,'o-',label='Train'); ax.semilogx(g.C,g.val_accuracy,'s--',label='Validation')
        ax.set(title=kernel,ylabel='Accuracy',ylim=(0,1.02)); ax.legend()
        ax=axs[1,col]; ax.semilogx(g.C,g.sv_fraction,'o-',color='tab:green'); ax.set(xlabel='C',ylabel='Support-vector fraction',ylim=(0,1.02))
    finish(fig,'svm_sweep')
    pairs=pd.read_csv(T/'pair_examples.csv'); images=np.load(T/'pair_images.npz')['images']
    for number,(pair,g) in enumerate(pairs.groupby('pair',sort=False)):
        fig,axs=plt.subplots(2,6,figsize=(7,2.05),constrained_layout=True)
        i,j=map(int,pair.split('-'))
        for row,source in enumerate(['support','test']):
            for col,(_,item) in enumerate(g[g.source==source].iterrows()):
                ax=axs[row,col]; ax.imshow(images[int(item.row_in_image_npz)]); ax.set(xticks=[],yticks=[])
                ax.set_title(f'T:{NAMES[int(item.true)][:4]} P:{NAMES[int(item.pred)][:4]}\nf={item.ovo_functional_score:+.2e}',fontsize=7)
                if col==0: ax.set_ylabel('Train SV' if source=='support' else 'Test',fontsize=7)
        fig.suptitle(f'{NAMES[i]} (+) versus {NAMES[j]} (−)',fontsize=9)
        finish(fig,f'pair_{number}')
    contingency=pd.read_csv(T/'cluster_contingency.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(3.35,3.0),constrained_layout=True)
    im=ax.imshow(contingency.to_numpy(),aspect='auto',cmap='Blues'); ax.set(xticks=range(10),xticklabels=[n[:4] for n in NAMES],yticks=range(len(contingency)),xlabel='True class',ylabel='Cluster')
    ax.tick_params(axis='x',rotation=60,labelsize=7); fig.colorbar(im,ax=ax,label='Test count',shrink=.8); finish(fig,'cluster_contingency')
    # Numerical interactions computed within paired initialisation seeds.
    interactions=[]
    for seed in [17,29]:
        a=table[table.seed==seed].set_index(['model','condition'])
        cnn_drop=float(a.loc[('cnn','clean'),'accuracy']-a.loc[('cnn','permuted'),'accuracy'])
        mlp_drop=float(a.loc[('mlp','clean'),'accuracy']-a.loc[('mlp','permuted'),'accuracy'])
        interactions.append({'seed':seed,'cnn_drop':cnn_drop,'mlp_drop':mlp_drop,'interaction':cnn_drop-mlp_drop})
    save_json(R/'paired_interactions.json',interactions)
    print('All figures generated from saved results.',flush=True)
if __name__=='__main__': main()
