"""Validate repaired Boolean geometry without changing original contour sites."""
from pathlib import Path
import sys, pickle, json
import numpy as np
import cv2
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'deps'))
from shapely import make_valid,intersects_xy
from shapely.geometry import Polygon,Point,LineString
import overlap_chords as rule
from topology_voronoi import inside_polygon
OUT=HERE/'full-14-10.20'


def main():
    data=pickle.load((OUT/'frames64.pkl').open('rb'))
    rng=np.random.default_rng(20261020)
    rows=[];tiles=[]
    for fi in (0,500,1400,2400,2850,2900):
        P=data['frames'][fi]['polys'];before=P.copy();repairs=[]
        plans,issues=rule.plan(P,repair_invalid=True,repairs=repairs)
        assert np.array_equal(P,before)
        count=0
        for repair in repairs:
            p=P[repair['a']];shape=rule.polygonal(make_valid(Polygon(p)))
            xy=rng.uniform(p.min(0)-1,p.max(0)+1,(1000,2))
            expected=np.array([inside_polygon(float(x),float(y),p) for x,y in xy])
            actual=intersects_xy(shape,xy[:,0],xy[:,1])
            assert np.array_equal(expected,actual)
            count+=len(xy)
        for plan in plans:
            points=plan['points']
            assert len(points)==plan['intersection_count']
            assert plan['segment_count']==len(points)-1
            for key in ('a','b'):
                p=P[plan[key]];ring=LineString(np.vstack([p,p[0]]))
                assert max(ring.distance(Point(q)) for q in points)<1e-6
            A,B=plan['part_a'],plan['part_b']
            assert A.intersection(B).area<1e-7
            assert A.union(B).symmetric_difference(plan['overlap']).area<1e-6
        rows.append(dict(frame=fi,unchanged_64_point_contours=True,
                         repaired_masks=len(repairs),parity_query_checks=count,
                         overlap_plans=len(plans),unresolved=len(issues),
                         area_conservation=True,all_intersections_on_original_contours=True))
        img=cv2.imread(str(OUT/f'preview_f{fi:04d}.png'))
        tile=cv2.resize(img,(1004,270),interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile,(0,0),(260,30),(30,30,30),-1)
        cv2.putText(tile,f'Frame {fi} / {fi/30:.2f} s',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
        if plans:
            plan=max(plans,key=lambda p:p['intersection_count'])
            if plan['intersection_count']>2:
                xy=plan['points'];lo=np.maximum(0,np.floor(xy.min(0)-25).astype(int))
                hi=np.minimum([img.shape[1],img.shape[0]],np.ceil(xy.max(0)+25).astype(int))
                crop=img[lo[1]:hi[1],lo[0]:hi[0]]
                crop=cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
                for i,p in enumerate(np.round((xy-lo)*4).astype(int)):
                    cv2.circle(crop,tuple(p),4,(0,165,255),-1)
                    cv2.putText(crop,str(i+1),tuple(p+[6,-6]),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,165,255),1,cv2.LINE_AA)
                cv2.imwrite(str(OUT/f'overlap_detail_f{fi:04d}.png'),crop)
    cv2.imwrite(str(OUT/'contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,6,2)]))
    (OUT/'geometry_validation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
