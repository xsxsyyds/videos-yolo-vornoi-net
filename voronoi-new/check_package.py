"""Integration checks for packaging, empty frames and original-frame equality."""
from pathlib import Path
import argparse,json,pickle,subprocess,sys
from _deps import configure_dependencies
configure_dependencies()
import cv2
import numpy as np
from render_video import compose
import overlap_chords

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference-run',type=Path,help='Original full-14-10.20 directory, optional')
    args=ap.parse_args()
    out=HERE/'tests'/'smoke';out.mkdir(parents=True,exist_ok=True)
    theta=np.arange(64)*2*np.pi/64
    circle=np.c_[np.cos(theta),np.sin(theta)]*25
    pair=np.asarray([circle+[65,60],circle+[95,60]],np.float32)
    plans,issues=overlap_chords.plan(pair,repair_invalid=True)
    assert len(plans)==1 and not issues and plans[0]['segment_count']==1
    frames=[]
    writer=cv2.VideoWriter(str(out/'input.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(160,120))
    assert writer.isOpened()
    for n in (0,1,2):
        polys=pair[:n].copy()
        item=dict(polys=polys,tid=np.arange(n),cls=np.zeros(n,dtype=int),num=np.arange(n)+1)
        frames.append(item)
        frame=np.full((120,160,3),45,np.uint8)
        writer.write(frame)
        before=polys.copy()
        result,stats,detail=compose(frame,item)
        assert np.array_equal(polys,before)
        assert stats['required_polyline_missing_pixels']==0
        if n<2: assert np.array_equal(result,frame)
        if n==2: assert np.any(result!=frame)
    writer.release()
    with (out/'frames64.pkl').open('wb') as f:
        pickle.dump(dict(frames=frames,meta=(160,120,30.,3,0,119)),f)
    subprocess.run([sys.executable,str(HERE/'render_video.py'),
        '--source',str(out/'input.mp4'),'--cache',str(out/'frames64.pkl'),
        '--out-dir',str(out/'output'),'--workers','2','--overwrite'],check=True)
    result=dict(synthetic_empty_single_overlap_frames=True,
                multiprocess_encode_decode=True,original_frame_checks=[])
    if args.reference_run:
        root=args.reference_run.resolve()
        data=pickle.load((root/'frames64.pkl').open('rb'))
        metadata=json.loads((root/'validation.json').read_text(encoding='utf-8'))
        cap=cv2.VideoCapture(metadata['source']);assert cap.isOpened()
        for fi in (0,500,1400,2400,2850,2900):
            cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
            ok,frame=cap.read();assert ok
            actual,stats,detail=compose(frame,data['frames'][fi])
            expected=cv2.imread(str(root/f'result_f{fi:04d}.png'))
            assert expected is not None
            different=int(np.count_nonzero(np.any(expected!=actual,axis=2)))
            assert different==0,(fi,different)
            result['original_frame_checks'].append(dict(frame=fi,different_pixels=different))
        cap.release()
    (HERE/'tests'/'package_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
