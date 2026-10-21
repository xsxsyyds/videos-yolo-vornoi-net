"""Reproduce tracking postprocessing used by 14-10.20.mp4, without inference."""
from pathlib import Path
import sys,pickle,json,time,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'full-14-10.20'
sys.path.insert(0,str(OUT))
import track_segment_snapshot as track

def main():
    d=pickle.load(open(HERE.parent/'_cache/14-10.20_pass1.pkl','rb'))
    reference=json.loads((HERE.parent/'_cache/14-10.20_tracks.json').read_text(encoding='utf-8'))
    t=time.perf_counter();tracks=d['tracks'];w,h,fps,nf,top,bot=d['meta']
    tracks,merged=track.stitch_tracks(tracks,max_gap=40,max_res=90.,max_angle=60.,border=(60,top+60,w-60,bot-60))
    track.fix_merged_masks(tracks,area_ratio=1.45,cover=.5,look=3,max_len=12)
    track.fix_overlap_conflicts(tracks,jump=.8,base_thr=.25,drop_conf=.6,center_ratio=.35)
    tracks={tid:tr for tid,tr in tracks.items() if len(tr['fi'])>=5}
    expected={int(tr['tid']):(tr['cls'],tr['num'],tr['first'],tr['last'],tr['frames']) for tr in reference['tracks']}
    actual={int(tid):(track.CLS_NAMES[tr['cls']],tr['num'],min(tr['fi']),max(tr['fi']),len(tr['fi'])) for tid,tr in tracks.items()}
    assert expected==actual,[(k,expected.get(k),actual.get(k)) for k in set(expected)|set(actual) if expected.get(k)!=actual.get(k)]
    print('Track identity/ranges/counts match original video JSON:',len(tracks),flush=True)
    per=track.smooth_tracks(tracks,nf,win=15,max_gap=45,poly_win=11,poly_outlier=2.2,
        shape_max=.06,clamp_tol=.03,poly_fill_gap=20,fill_max_body=2.5)
    frames=[]
    for fi in range(nf):
        items=sorted(per.get(fi,[]),key=lambda it:it['tid'])
        items=[it for it in items if it['poly'] is not None and np.isfinite(it['poly']).all()]
        assert all(np.asarray(it['poly']).shape==(64,2) for it in items)
        frames.append(dict(polys=np.asarray([it['poly'] for it in items],np.float32).reshape(-1,64,2),
            tid=np.array([it['tid'] for it in items]),cls=np.array([it['cls'] for it in items]),
            num=np.array([it['num'] for it in items])))
    with (OUT/'frames64.pkl').open('wb') as f:pickle.dump(dict(frames=frames,meta=d['meta']),f,protocol=5)
    metadata=dict(frames=nf,width=w,height=h,fps=fps,vertices=64,tracks=len(tracks),
        matches_original_track_json=True,preparation_seconds=time.perf_counter()-t,
        max_objects=max(len(f['polys']) for f in frames),mean_objects=float(np.mean([len(f['polys']) for f in frames])),
        tracking_source_sha256=hashlib.sha256((OUT/'track_segment_snapshot.py').read_bytes()).hexdigest())
    (OUT/'preparation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(metadata,flush=True)

if __name__=='__main__':main()
