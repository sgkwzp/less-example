"""Complete paired development analysis; no candidate/threshold selection."""
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
from aem_scarcity_diagnostic import metrics_boot,annotation_frames,contrast,independent_auc,auc
import aem_cross_route_gpu as io

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'aem_support_route_runtime'

def main():
    reg=io.read(OUT/'registered.json');done=io.read(OUT/'gpu_complete.json');assert done['complete']
    assert done['registration_sha']==io.sha(OUT/'registered.json')
    assert all(io.sha(ROOT/p)==h for p,h in reg['hashes'].items())
    assert all(io.sha(OUT/p)==h for p,h in done['hashes'].items())
    ann={v['video_id']:v['labels'] for v in io.read(ROOT/'code/AEM/data/egoper/annotation.json')['coffee']['segments']}
    ids=reg['jobs'][0]['test'];assert len(ids)==38 and all(j['test']==ids for j in reg['jobs'])
    rng=np.random.default_rng(20261005);weights=np.vstack([np.ones(38,int),rng.multinomial(38,np.full(38,1/38),2000)])
    results={};arrays=[];mechanism=[];verify=dict(files=0,max_auc_error=0.,max_explicit_bootstrap_error=0.,background_changes=0)
    for seed in [0,1]:
        rows=[];route_counts={m:{kind:[0,0] for kind in ['known_normal','known_error']} for m in ['N','L']}
        segment_counts={m:dict(segments=0,singletons=0) for m in ['N','L']}
        for vid in ids:
            n=dict(np.load(HERE/'aem_cross_route_runtime/replay/C'/f's{seed}_KK'/(vid+'.npz')))
            l=dict(np.load(OUT/'scores'/f's{seed}'/(vid+'.npz')))
            gt,typ=annotation_frames(ann[vid],len(n['labels']));row=dict(video=vid,gt=gt,types=typ,y=typ>0)
            assert np.array_equal(n['pred_action']==0,l['pred_action']==0)
            source=np.load(OUT/'routes'/f's{seed}'/(vid+'.npz'))
            for k in ['segments','segment_action_pred','pred_action']:assert np.array_equal(l[k],source[k])
            for m,z in [('N',n),('L',l)]:
                assert np.array_equal(gt,z['gt_action']) and np.array_equal(typ>0,z['labels']>0)
                rec=np.empty(len(gt),np.float32);pred=np.empty(len(gt),int);cover=np.zeros(len(gt),int)
                for (a,b),c,s in zip(z['segments'],z['segment_action_pred'],z['segment_action_sim']):
                    assert 0<=a<=b<len(gt);rec[a:b+1]=1-float(s);pred[a:b+1]=c;cover[a:b+1]+=1
                assert (cover==1).all() and np.array_equal(pred,z['pred_action'])
                assert np.array_equal(rec,z['raw_all']) and np.array_equal(rec*(pred!=0),z['scores'])
                row[m]=z['scores']
                segment_counts[m]['segments']+=len(z['segments'])
                segment_counts[m]['singletons']+=int((z['segments'][:,0]==z['segments'][:,1]).sum())
                for kind,mask in [('known_normal',(gt>0)&(typ==0)),('known_error',(gt>0)&(typ>0))]:
                    route_counts[m][kind][0]+=int(((pred==gt)&mask).sum());route_counts[m][kind][1]+=int(mask.sum())
            rows.append(row);verify['files']+=1
        acc={m:{k:c/t for k,(c,t) in d.items()} for m,d in route_counts.items()}
        mechanism.append(acc)
        for smooth in [True,False]:
            wr=[{**r,**{m:median_filter(r[m],size=21,mode='nearest') if smooth else r[m] for m in ['N','L']}} for r in rows]
            met,arr,err=metrics_boot(wr,['N','L'],weights);verify['max_auc_error']=max(verify['max_auc_error'],err)
            ix=np.repeat(np.arange(38),weights[1]);yy=np.concatenate([wr[i]['y'] for i in ix]);ss=np.concatenate([wr[i]['L'] for i in ix])
            err=abs(independent_auc(yy,ss)-arr['L']['whole'][1]);assert err<1e-12
            verify['max_explicit_bootstrap_error']=max(verify['max_explicit_bootstrap_error'],err)
            y=np.concatenate([r['y'] for r in wr]);types=np.concatenate([r['types'] for r in wr])
            bytype={};normal={}
            for m in ['N','L']:
                s=np.concatenate([r[m] for r in wr]);assert independent_auc(y,np.full(len(y),.5))==.5
                bytype[m]={str(t):dict(positive_frames=int((types==t).sum()),auc=auc(y[(types==0)|(types==t)],s[(types==0)|(types==t)])) for t in np.unique(types) if t>0}
                nv=[r for r in wr if '_normal_' in r['video']];ns=np.concatenate([r[m] for r in nv])
                normal[m]=dict(videos=len(nv),frames=len(ns),fraction_score_ge_05=float((ns>=.5).mean()),quantiles=np.quantile(ns,[.5,.9,.95,.99]).tolist())
            key=f's{seed}_'+('median21' if smooth else 'raw')
            results[key]=dict(metrics=met,contrast=contrast(arr,'L','N'),routing=acc,segments=segment_counts,by_error_type=bytype,normal=normal)
            if smooth:arrays.append(arr)
    mean={m:{k:(arrays[0][m][k]+arrays[1][m][k])/2 for k in ['whole','known','within']} for m in ['N','L']}
    delta=contrast(mean,'L','N');meanacc={m:{k:np.mean([a[m][k] for a in mechanism]).item() for k in ['known_normal','known_error']} for m in ['N','L']}
    gates=dict(mean_all_positive=all(delta[k]['delta_pp']>0 for k in delta),
        main_ci_positive=all(delta[k]['ci95_pp'][0]>0 for k in ['known','within']),
        each_seed_main_nonnegative=all(results[f's{s}_median21']['contrast'][k]['delta_pp']>=0 for s in [0,1] for k in ['known','within']),
        above_constant=all(mean['L'][k][0]>.5 for k in mean['L']),
        routing_normal_improves=meanacc['L']['known_normal']>meanacc['N']['known_normal'],
        routing_error_nonnegative=meanacc['L']['known_error']>=meanacc['N']['known_error'])
    results['seedmean']=dict(metrics={m:{k:dict(auc=float(v[0])) for k,v in d.items()} for m,d in mean.items()},contrast=delta,routing=meanacc)
    results['gates']=gates;results['continue']=all(gates.values());results['verification']=verify
    io.save(OUT/'results.json',results)
    io.save(OUT/'transfer_verification.json',dict(files=len(done['hashes']),all_hashes_match=True))
    lines=['# K2 示范顺序路由：开发实验结果','',
        '结论：'+('通过预定开发门槛，可另行设计扩展验证。' if results['continue'] else '未通过预定门槛，停止当前线性模板候选，不调参续跑。'),'',
        '只用 coffee 完整38条测试视频、固定 K2 支持种子0/1；76次运行不是76独立视频。冻结 AEM K2 模型、保留原背景掩码；L用各自两条正常示范的实际顺序，替换正步骤类别和边界。N为当前环境原生路由。两者均原生门控＋median21。', '',
        '| 运行 | 整体 AUC% | GT>0 AUC% | 步骤内 AUC% |','|---|---:|---:|---:|']
    for name in ['s0_median21','s1_median21','seedmean']:
        for m in ['N','L']:
            d=results[name]['metrics'][m];lines.append('| '+name+' '+m+' | '+' | '.join(f"{100*d[k]['auc']:.3f}" for k in ['whole','known','within'])+' |')
    lines+=['','两种子均值差 L−N，2000次共享视频配对 bootstrap：','', '| 指标 | 差值 pp | 95%CI pp |','|---|---:|---:|']
    for k,d in delta.items():lines.append(f"| {k} | {d['delta_pp']:+.3f} | [{d['ci95_pp'][0]:+.3f}, {d['ci95_pp'][1]:+.3f}] |")
    lines+=['','机制读出（GT已知正步骤帧，两种子准确率均值）：','', '| 帧组 | N% | L% |','|---|---:|---:|']
    for k in meanacc['N']:lines.append(f"| {k} | {100*meanacc['N'][k]:.3f} | {100*meanacc['L'][k]:.3f} |")
    lines+=['','门槛：'+str(gates),'',
        f"校验：76份输出与服务器SHA一致；全部原路由重评分最大误差 {max(x['control_error'] for x in done['checks']):.3g}；新路由覆盖/背景掩码/逐帧分数重建全部一致；AUC独立核验误差 {verify['max_auc_error']:.3g}，显式bootstrap核验 {verify['max_explicit_bootstrap_error']:.3g}。常数对照AUC50。",'',
        f"GPU评分核心耗时 {done['core_seconds']:.2f} 秒，不含模型初始化、编译、传输或分析。没有训练、VLM、新增标注或保留任务评分。",'',
        '本实验使用公开支持步骤标注、原生预训练视觉特征和 AEM 文本/模型；仅路由模板来自K条支持，不能主张总知识仅K或无需已有标注。固定背景掩码限制了可恢复的误差。所有结果来自已多轮使用的开发任务，不是独立确认。原始分数、按错误类型、正常视频分数/固定0.5描述性误报与单种子指标见机器结果；未校准部署阈值。', '',
        '证据：[协议](aem-support-route-protocol.md)、`aem_support_route_runtime/{registered,routes_complete,gpu_complete,results,transfer_verification}.json`。']
    (HERE/'aem-support-route-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()
