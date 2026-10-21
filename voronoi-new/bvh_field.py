"""Exact continuous segment nearest-neighbor queries with an AABB BVH.

No contour subsampling or distance-transform approximation. This computes only
the nearest site, NOT all per-agent fields. Tie breaking is lowest edge index
using stable direct float64 distances; old BLAS bit patterns are not preserved.
"""
import time
from pathlib import Path
import sys
import numpy as np
from _deps import configure_dependencies
configure_dependencies()
from numba import njit,prange,set_num_threads


def build_tree(edges,leaf_size=8):
    lo=edges.min(1);hi=edges.max(1);centers=(lo+hi)*.5
    boxes=[];children=[];ranges=[];order=[]
    def visit(ids):
        node=len(boxes)
        low=lo[ids].min(0);high=hi[ids].max(0)
        boxes.append(np.r_[low,high]);children.append([-1,-1]);ranges.append([0,0])
        if len(ids)<=leaf_size:
            start=len(order);order.extend(ids);ranges[node]=[start,len(order)]
        else:
            axis=int(np.argmax(np.ptp(centers[ids],axis=0)))
            ids=ids[np.argsort(centers[ids,axis],kind='stable')]
            mid=len(ids)//2
            children[node]=[visit(ids[:mid]),visit(ids[mid:])]
        return node
    visit(np.arange(len(edges)))
    return np.asarray(boxes,np.float64),np.asarray(children,np.int32),np.asarray(ranges,np.int32),np.asarray(order,np.int32)


@njit(cache=True)
def build_tree_compiled(edges,leaf_size=8):
    e=len(edges)
    boxes=np.empty((2*e,4),np.float64)
    children=np.full((2*e,2),-1,np.int32)
    ranges=np.zeros((2*e,2),np.int32)
    order=np.arange(e,dtype=np.int32)
    centers=(edges[:,0]+edges[:,1])*.5
    ranges[0,1]=e
    node=0;count=1
    while node<count:
        start,end=ranges[node]
        xmin=ymin=cxmin=cymin=np.inf
        xmax=ymax=cxmax=cymax=-np.inf
        for j in range(start,end):
            ei=order[j]
            cxmin=min(cxmin,centers[ei,0]);cxmax=max(cxmax,centers[ei,0])
            cymin=min(cymin,centers[ei,1]);cymax=max(cymax,centers[ei,1])
            for v in range(2):
                xmin=min(xmin,edges[ei,v,0]);xmax=max(xmax,edges[ei,v,0])
                ymin=min(ymin,edges[ei,v,1]);ymax=max(ymax,edges[ei,v,1])
        boxes[node,0]=xmin;boxes[node,1]=ymin
        boxes[node,2]=xmax;boxes[node,3]=ymax
        if end-start>leaf_size:
            axis=0 if cxmax-cxmin>=cymax-cymin else 1
            ids=order[start:end].copy()
            ranked=np.argsort(centers[ids,axis],kind='mergesort')
            order[start:end]=ids[ranked]
            mid=(start+end)//2
            children[node,0]=count;children[node,1]=count+1
            ranges[count,0]=start;ranges[count,1]=mid
            ranges[count+1,0]=mid;ranges[count+1,1]=end
            count+=2
        node+=1
    return boxes[:count],children[:count],ranges[:count],order


@njit(cache=True,inline='always')
def box_d2(x,y,b):
    dx=max(b[0]-x,0.,x-b[2]);dy=max(b[1]-y,0.,y-b[3])
    return dx*dx+dy*dy


@njit(cache=True,parallel=True,fastmath=False)
def query(points,edges,boxes,children,ranges,order):
    prim=np.empty(len(points),np.int32)
    dist=np.empty(len(points),np.float64)
    evaluations=np.zeros(len(points),np.int32)
    for i in prange(len(points)):
        x,y=points[i]
        stack=np.empty(128,np.int32);stack[0]=0;top=1
        best=np.inf;winner=2147483647;count=0
        while top:
            top-=1;node=stack[top]
            # A small conservative numerical guard avoids pruning an equal
            # distance through floating-point roundoff at a bounding box.
            if box_d2(x,y,boxes[node])>best+1e-10*max(1.,best):continue
            left,right=children[node]
            if left<0:
                start,end=ranges[node]
                for j in range(start,end):
                    e=order[j]
                    ax,ay=edges[e,0];bx,by=edges[e,1]
                    vx,vy=bx-ax,by-ay;wx,wy=x-ax,y-ay
                    vv=vx*vx+vy*vy
                    t=min(1.,max(0.,(wx*vx+wy*vy)/vv)) if vv>1e-12 else 0.
                    dx,dy=wx-t*vx,wy-t*vy
                    d=dx*dx+dy*dy
                    count+=1
                    if d<best or (d==best and e<winner):best=d;winner=e
            else:
                dl=box_d2(x,y,boxes[left]);dr=box_d2(x,y,boxes[right])
                if dl<dr:
                    stack[top]=right;stack[top+1]=left
                else:
                    stack[top]=left;stack[top+1]=right
                top+=2
        prim[i]=winner;dist[i]=np.sqrt(best);evaluations[i]=count
    return prim,dist,evaluations


def build_field(polys,region,scale=.25,workers=4):
    set_num_threads(workers)
    start=time.perf_counter()
    x0,y0,x1,y1=region
    w=max(2,round((x1-x0)*scale));h=max(2,round((y1-y0)*scale))
    xs=x0+(np.arange(w)+.5)/scale;ys=y0+(np.arange(h)+.5)/scale
    X,Y=np.meshgrid(xs,ys)
    points=np.c_[X.ravel(),Y.ravel()]
    A=np.asarray(polys,np.float32)
    edges=np.stack([A,np.roll(A,-1,axis=1)],2).reshape(-1,2,2).astype(np.float64)
    tree=build_tree_compiled(edges)
    built=time.perf_counter()
    prim,D,counts=query(points,edges,*tree)
    m,k=A.shape[:2];e2a=np.repeat(np.arange(m),k)
    return dict(prim=prim.reshape(h,w),D=D.reshape(h,w),Ag=e2a[prim].reshape(h,w),
        edges=edges,e2a=e2a,M=m,K=k,E=len(edges),W=w,H=h,xs=xs,ys=ys,X=X,Y=Y,
        Pflat=points,polys=A,region=region,scale=scale,
        tree_ms=(built-start)*1000,query_ms=(time.perf_counter()-built)*1000,
        mean_edges_tested=float(counts.mean()),max_edges_tested=int(counts.max()))
