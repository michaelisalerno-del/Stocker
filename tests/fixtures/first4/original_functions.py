import numpy as np

def rank(a,extra=None):
 valid=np.isfinite(a).all(2)&(a[:,:,2]>=5.5)
 if extra is not None:valid&=extra
 order=np.argsort(-np.where(valid,a[:,:,1],-np.inf),axis=1,kind="stable");rr=np.empty(order.shape,np.int16)
 np.put_along_axis(rr,order,np.broadcast_to(np.arange(1,a.shape[1]+1),order.shape),axis=1);rr[~valid]=32767
 return rr

def valid(a):return len(a)>0 and np.isfinite(a).all() and (a>0).all() and (a[:,1]>=np.maximum(a[:,0],a[:,3])).all() and (a[:,2]<=np.minimum(a[:,0],a[:,3])).all()

def excursion(a,s):
 if not valid(a) or not np.isfinite(s) or s<=0:return (np.nan,np.nan)
 up=max(0,(a[:,1].max()/s-1)*10000);dn=max(0,(1-a[:,2].min()/s)*10000)
 return max(up,dn)/100,(up+dn)/100

def select(stream,cutoff,cap):
 accepted=[];statuses=[]
 for event_id,clock in stream:
  reason='DAILY_CAP' if len(accepted)>=4 else 'RESERVATION' if clock<cutoff and len(accepted)>=cap else 'ACCEPT'
  if reason=='ACCEPT':accepted.append(event_id)
  statuses.append((event_id,reason,len(accepted) if reason=='ACCEPT' else np.nan))
 return statuses
