"""Experimental solid-polygon exterior diagram, with explicit cell topology.

Changes semantics intentionally: polygon interiors are occupied, and roots use
the entire polygon. Raster output is NOT intended to equal the old buggy output.
The old grid resolution limits detection of tiny regions; this is not an exact
analytic segment-Voronoi implementation.
"""
import math
import time
import numpy as np
import cv2
import fast_voronoi as fast
import bvh_field
import overlap_chords
from numba import njit,prange


@njit(cache=True)
def polygon_distance(x,y,p):
    best=np.inf; gx=gy=0.
    for j in range(len(p)):
        ax,ay=float(p[j,0]),float(p[j,1])
        bx,by=float(p[(j+1)%len(p),0]),float(p[(j+1)%len(p),1])
        vx,vy=bx-ax,by-ay
        wx,wy=x-ax,y-ay
        vv=vx*vx+vy*vy
        t=min(1.,max(0.,(wx*vx+wy*vy)/vv)) if vv>1e-12 else 0.
        dx,dy=wx-t*vx,wy-t*vy
        d=dx*dx+dy*dy
        if d<best:best=d;gx=dx;gy=dy
    dist=math.sqrt(best)
    if dist>1e-12:gx/=dist;gy/=dist
    return dist,gx,gy


@njit(cache=True)
def inside_polygon(x,y,p):
    inside=False
    for j in range(len(p)):
        ax,ay=float(p[j,0]),float(p[j,1])
        bx,by=float(p[(j+1)%len(p),0]),float(p[(j+1)%len(p),1])
        if (ay>y)!=(by>y):
            if x<(bx-ax)*(y-ay)/(by-ay)+ax:inside=not inside
    return inside


@njit(cache=True)
def occupy_grid(owner,polys,xs,ys,bounds):
    out=owner.copy()
    count=np.zeros(owner.shape,np.int16)
    for y in range(len(ys)):
        for x in range(len(xs)):
            for a in range(len(polys)):
                if xs[x]<bounds[a,0] or xs[x]>bounds[a,2] or ys[y]<bounds[a,1] or ys[y]>bounds[a,3]:continue
                if inside_polygon(xs[x],ys[y],polys[a]):
                    if count[y,x]==0:out[y,x]=a
                    count[y,x]+=1
    return out,count


@njit(cache=True)
def root(x0,y0,x1,y1,pa,pb):
    g0=polygon_distance(x0,y0,pa)[0]-polygon_distance(x0,y0,pb)[0]
    g1=polygon_distance(x1,y1,pa)[0]-polygon_distance(x1,y1,pb)[0]
    if g0*g1>0:return (x0+x1)*.5,(y0+y1)*.5,False
    lo=0.;hi=1.
    for _ in range(20):
        t=(lo+hi)*.5
        x=x0+t*(x1-x0);y=y0+t*(y1-y0)
        g=polygon_distance(x,y,pa)[0]-polygon_distance(x,y,pb)[0]
        if g*g0>0:lo=t
        else:hi=t
    t=(lo+hi)*.5
    return x0+t*(x1-x0),y0+t*(y1-y0),True


@njit(cache=True,parallel=True)
def crossings(owner,polys,xs,ys):
    h,w=owner.shape
    points=np.full((h,w,3,2),np.nan,np.float64)
    valid=np.zeros((h,w,3),np.bool_)
    for y in prange(h):
        for x in range(w):
            for d in range(3):
                yy=y+(d!=0);xx=x+(d!=1)
                if xx>=w or yy>=h:continue
                a,b=owner[y,x],owner[yy,xx]
                if a==b:continue
                px,py,ok=root(xs[x],ys[y],xs[xx],ys[yy],polys[a],polys[b])
                points[y,x,d,0]=px;points[y,x,d,1]=py
                valid[y,x,d]=ok
    return points,valid


@njit(cache=True)
def junction(x,y,polys,a,b,c,xlo,ylo,xhi,yhi):
    for _ in range(50):
        da,ax,ay=polygon_distance(x,y,polys[a])
        db,bx,by=polygon_distance(x,y,polys[b])
        dc,cx,cy=polygon_distance(x,y,polys[c])
        f,g=da-db,da-dc
        if max(abs(f),abs(g))<1e-7:return x,y,True
        j00,j01=ax-bx,ay-by
        j10,j11=ax-cx,ay-cy
        det=j00*j11-j01*j10
        if abs(det)<1e-12:break
        dx=(f*j11-j01*g)/det
        dy=(j00*g-f*j10)/det
        # Nearest-feature changes make the equations nonsmooth. Use a bounded
        # damped Newton step instead of accepting a jump outside the cell.
        accepted=False;step=1.
        for ls in range(20):
            nx=x-step*dx;ny=y-step*dy
            if xlo<=nx<=xhi and ylo<=ny<=yhi:
                nda=polygon_distance(nx,ny,polys[a])[0]
                ndb=polygon_distance(nx,ny,polys[b])[0]
                ndc=polygon_distance(nx,ny,polys[c])[0]
                if (nda-ndb)**2+(nda-ndc)**2 < f*f+g*g:
                    x=nx;y=ny;accepted=True;break
            step*=.5
        if not accepted:break
    return x,y,False


@njit(cache=True)
def assemble(owner,polys,xs,ys,points,valid):
    h,w=owner.shape
    lines=np.empty((6*(h-1)*(w-1),2,2),np.float64)
    pairs=np.empty((len(lines),2),np.int32)
    n=0;failed=0;unbracketed=0
    for y in range(h-1):
        for x in range(w-1):
            o0,o1,o2,o3=owner[y,x],owner[y,x+1],owner[y+1,x+1],owner[y+1,x]
            if o0==o1 and o1==o2 and o2==o3:continue
            for tri in range(2):
                if tri==0 and o0==o1 and o1==o2:continue
                if tri==1 and o0==o2 and o2==o3:continue
                # Shared grid-edge root cache, including the cell diagonal.
                if tri==0:
                    ey=np.array([y,y,y]);ex=np.array([x,x+1,x]);ed=np.array([0,1,2])
                    labels=np.array([owner[y,x],owner[y,x+1],owner[y+1,x+1]])
                else:
                    ey=np.array([y,y+1,y]);ex=np.array([x,x,x]);ed=np.array([2,0,1])
                    labels=np.array([owner[y,x],owner[y+1,x+1],owner[y+1,x]])
                cross=np.empty((3,2));cp=np.empty((3,2),np.int32)
                nc=0
                for j in range(3):
                    a,b=labels[j],labels[(j+1)%3]
                    if a==b:continue
                    yy,xx,d=ey[j],ex[j],ed[j]
                    if not valid[yy,xx,d]:
                        unbracketed+=1
                        continue
                    cross[nc]=points[yy,xx,d]
                    cp[nc,0]=min(a,b);cp[nc,1]=max(a,b)
                    nc+=1
                if nc==2 and cp[0,0]==cp[1,0] and cp[0,1]==cp[1,1]:
                    lines[n,0]=cross[0];lines[n,1]=cross[1];pairs[n]=cp[0];n+=1
                elif nc==3:
                    jx,jy=cross[:,0].mean(),cross[:,1].mean()
                    jx,jy,ok=junction(jx,jy,polys,labels[0],labels[1],labels[2],xs[x],ys[y],xs[x+1],ys[y+1])
                    if not ok:
                        failed+=1
                        # Keep this uncertainty visible in metadata; no arbitrary
                        # shortcut is drawn through a failed triple junction.
                        continue
                    for j in range(3):
                        lines[n,0]=cross[j];lines[n,1,0]=jx;lines[n,1,1]=jy
                        pairs[n]=cp[j];n+=1
    return lines[:n],pairs[:n],failed,unbracketed


def compute(polys,region,scale=.25,backend='bvh',repair_invalid=False):
    start=time.perf_counter()
    F=bvh_field.build_field(polys,region,scale) if backend=='bvh' else fast.build_field(polys,region,scale)
    tf=time.perf_counter()
    bounds=np.c_[polys.min(1),polys.max(1)].astype(np.float64)
    owner,occupied=occupy_grid(F['Ag'],polys,F['xs'],F['ys'],bounds)
    repairs=[]
    chord_plans,chord_issues=overlap_chords.plan(polys,repair_invalid=repair_invalid,repairs=repairs)
    owner=overlap_chords.update_owner(owner,F['xs'],F['ys'],chord_plans)
    F['overlap_chords']=chord_plans
    F['overlap_issues']=chord_issues
    F['polygon_repairs']=repairs
    ti=time.perf_counter()
    pts,valid=crossings(owner,polys,F['xs'],F['ys'])
    tc=time.perf_counter()
    lines,pairs,failed,unbracketed=assemble(owner,polys,F['xs'],F['ys'],pts,valid)
    return F,owner,lines,pairs,dict(field_ms=(tf-start)*1000,
        topology_ms=(time.perf_counter()-tf)*1000,
        occupancy_ms=(ti-tf)*1000,crossings_ms=(tc-ti)*1000,
        assembly_ms=(time.perf_counter()-tc)*1000,
        failed_junctions=int(failed),unbracketed_crossings=int(unbracketed),
        overlap_chord_count=len(chord_plans),overlap_issue_count=len(chord_issues),
        repaired_boolean_geometries=len(repairs),
        occupied_grid_points=int((occupied>0).sum()),overlap_grid_points=int((occupied>1).sum()))


def rasterize(F,polys,lines,ss=3,thick_img=2):
    h,w=F['H']*ss,F['W']*ss
    sc=F['scale']*ss
    offset=np.array(F['region'][:2])
    mask=np.zeros((h,w),np.uint8)
    q=np.round((lines-offset)*sc).astype(np.int32)
    if len(q):cv2.polylines(mask,list(q),False,255,max(1,round(thick_img*sc)))
    occupied=np.zeros_like(mask)
    # Raster clipping implements an explicitly exterior-only display. Overlap
    # ownership is unresolved physically; no interior separators are invented.
    for p in polys:
        cv2.fillPoly(occupied,[np.round((p-offset)*sc).astype(np.int32)],255)
    mask[occupied>0]=0
    plans=F.get('overlap_chords')
    if plans is None:plans,_=overlap_chords.plan(polys)
    mask=overlap_chords.draw_chords(F,mask,plans,ss=ss,thick_img=thick_img)
    return mask,occupied
