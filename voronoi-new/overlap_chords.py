"""User-defined overlap interface: ordered polyline through ALL intersections.

This is a geometric partition convention, not a claim of equal distances.
Connect 1-2-3-... without closing the chain. Ordering uses contour arc length,
cut at the largest inter-crossing gap; verify agreement on both contours.
Keep original contours and report unsupported/ambiguous geometry explicitly.
"""
from pathlib import Path
import sys
import numpy as np
import cv2
from _deps import configure_dependencies
configure_dependencies()
from shapely.geometry import Polygon,LineString,Point
from shapely.ops import unary_union,split
from shapely import contains_xy,intersects_xy,make_valid,is_valid_reason


def polygonal(g):
    if g.geom_type in ('Polygon','MultiPolygon'):return g
    return unary_union([part for part in getattr(g,'geoms',[]) if part.geom_type in ('Polygon','MultiPolygon')])


def ordered_crossings(shape,points,boundary=None):
    if boundary is None:boundary=shape.exterior
    values=np.array([boundary.project(p) for p in points])
    order=np.argsort(values,kind='stable')
    distances=values[order]
    gaps=np.diff(np.r_[distances,distances[0]+boundary.length])
    cut=int(np.argmax(gaps))
    order=np.roll(order,-(cut+1))
    xy=np.array([[points[i].x,points[i].y] for i in order])
    # Canonical chain direction is independent of polygon winding/start index.
    if tuple(xy[-1])<tuple(xy[0]):xy=xy[::-1]
    return xy


def plan(polys,repair_invalid=False,repairs=None):
    shapes=[Polygon(np.asarray(p,np.float64)) for p in polys]
    boundaries=[s.exterior for s in shapes]
    if repair_invalid:
        for a,s in enumerate(shapes):
            if not s.is_valid:
                # Linework make_valid nodes the original edges and uses the
                # even-odd interior rule. It does not simplify/resample points.
                fixed=polygonal(make_valid(s))
                if fixed.is_valid and not fixed.is_empty:
                    shapes[a]=fixed
                    if repairs is not None:
                        repairs.append(dict(a=a,status='repaired_boolean_geometry',
                            reason=is_valid_reason(s),geometry_type=fixed.geom_type))
    result=[];issues=[]
    for a,A in enumerate(shapes):
        if not A.is_valid:
            issues.append(dict(a=a,status='invalid_polygon'))
            continue
        for b in range(a+1,len(shapes)):
            B=shapes[b]
            if not B.is_valid:continue
            al,ab,ar,at=A.bounds;bl,bb,br,bt=B.bounds
            if ar<bl or br<al or at<bb or bt<ab:continue
            overlap=A.intersection(B)
            if overlap.area<=1e-9:continue
            cross=A.boundary.intersection(B.boundary)
            if cross.geom_type=='MultiPoint':
                points=list(cross.geoms)
            elif cross.geom_type=='Point':points=[cross]
            else:points=[]
            info=dict(a=a,b=b,overlap_area=overlap.area,intersection_count=len(points))
            if len(points)<2:
                issues.append(dict(**info,status='needs_rule_for_crossings_or_containment',geometry_type=cross.geom_type))
                continue
            points=sorted(points,key=lambda p:(p.x,p.y))
            xy=ordered_crossings(A,points,boundaries[a])
            other=ordered_crossings(B,points,boundaries[b])
            if not np.allclose(xy,other,rtol=0,atol=1e-7):
                issues.append(dict(**info,status='contour_order_disagrees'));continue
            info['selection']='all_intersections_in_contour_order'
            info['all_intersections']=xy.tolist()
            info['segment_count']=len(xy)-1
            chord=LineString(xy)
            if not chord.is_simple:
                issues.append(dict(**info,status='self_crossing_interface'));continue
            info['chord_fully_inside_overlap']=bool(overlap.buffer(1e-8).covers(chord))
            triples=[c for c,C in enumerate(shapes) if c not in (a,b) and C.is_valid
                     and overlap.intersection(C).area>1e-9]
            if triples:
                issues.append(dict(**info,status='triple_overlap',others=triples));continue
            # Extend only the first and last segments. The full polyline is
            # the splitter; independently applying each segment's half-plane
            # would overwrite prior decisions around bends.
            first=xy[1]-xy[0];first/=np.linalg.norm(first)
            last=xy[-1]-xy[-2];last/=np.linalg.norm(last)
            reach=max(ar-al,at-ab,br-bl,bt-bb,1.)*4
            extended=LineString(np.vstack([xy[0]-first*reach,xy,xy[-1]+last*reach]))
            parts=[polygonal(p) for p in split(overlap,extended).geoms if p.area>1e-12]
            # Each half inherits the object whose exclusive area it borders.
            # Boundary overlap tolerances absorb polygon Boolean roundoff.
            exA=A.difference(B);exB=B.difference(A)
            def score(piece,exclusive):
                if piece.is_empty or exclusive.is_empty:return 0.
                return piece.boundary.intersection(exclusive.boundary.buffer(1e-7)).length
            a_parts=[];b_parts=[];ambiguous=False
            for piece in parts:
                sa,sb=score(piece,exA),score(piece,exB)
                if abs(sa-sb)<1e-7:ambiguous=True;break
                (a_parts if sa>sb else b_parts).append(piece)
            if ambiguous:
                issues.append(dict(**info,status='ambiguous_side_assignment'));continue
            partA,partB=unary_union(a_parts),unary_union(b_parts)
            info['positive_area_both_sides']=partA.area>1e-12 and partB.area>1e-12
            result.append(dict(**info,status='chord',points=xy,overlap=overlap,
                               part_a=partA,part_b=partB))
    return result,issues


def serializable(plans,issues):
    return dict(chords=[{k:(v.tolist() if isinstance(v,np.ndarray) else v)
                        for k,v in p.items() if k not in ('overlap','part_a','part_b')}
                       for p in plans],unresolved=issues)


def update_owner(owner,xs,ys,plans):
    out=owner.copy()
    for p in plans:
        lo,bot,hi,top=p['overlap'].bounds
        ix=np.flatnonzero((xs>=lo)&(xs<=hi));iy=np.flatnonzero((ys>=bot)&(ys<=top))
        if not len(ix) or not len(iy):continue
        X,Y=np.meshgrid(xs[ix],ys[iy]);block=out[np.ix_(iy,ix)]
        for key,label in (('part_a',p['a']),('part_b',p['b'])):
            block[intersects_xy(p[key],X,Y)]=label
        # Points exactly on the chord inherit the lower-index object.
        on=intersects_xy(LineString(p['points']),X,Y)&intersects_xy(p['overlap'],X,Y)
        block[on]=p['a']
        out[np.ix_(iy,ix)]=block
    return out


def draw_chords(F,mask,plans,ss=3,thick_img=2):
    out=mask.copy();sc=F['scale']*ss
    offset=np.array(F['region'][:2]);thick=max(1,round(thick_img*sc))
    for p in plans:
        # Consecutive intersections form one open polyline, never a shortcut
        # from the first point to the last or a closing last-to-first segment.
        q=np.round((p['points']-offset)*sc).astype(np.int32)
        cv2.polylines(out,[q],False,255,thick,cv2.LINE_8)
    return out
