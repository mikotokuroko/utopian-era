"""Generate seamless planetary materials and ray-traced atlas portraits."""
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image
A=Path(__file__).parent/'docs/assets'
W,H=1536,768
lat,lon=np.mgrid[0:H,0:W]; lat=(.5-lat/(H-1))*np.pi;lon=lon/W*2*np.pi
p=np.array([np.cos(lat)*np.cos(lon),np.sin(lat),np.cos(lat)*np.sin(lon)]).astype('f')
def unit(v): return v/np.linalg.norm(v)
for k in range(3):
 rng=np.random.default_rng(41+k*38)
 grid=rng.uniform(-1,1,(53,53,53)).astype('f')
 def noise(v,f,o=0):return map_coordinates(grid,v*f+25+o,order=1,mode='grid-wrap',prefilter=False)
 def fb(v,f,n):return sum(noise(v,f*2**i,i*2)*.53**i for i in range(n))
 q=p+np.array([fb(p,2,5),fb(p,2.2,5),fb(p,1.9,5)])*.18
 e=fb(q,2.4,8); fine=fb(q,55,4)
 ridge=(1-np.abs(fb(q,9,6))).clip(0,1)**6
 land=np.clip((e+.06)*13,0,1)
 ocean,forest,peak,cloud=[([14,42,52],[64,97,68],[185,188,147],[211,226,214]),([18,34,55],[85,119,142],[202,223,234],[225,234,247]),([45,29,36],[159,94,62],[234,185,130],[237,204,174])][k]
 def mix(a,b,w):return np.array(a)*(1-w[...,None])+np.array(b)*w[...,None]
 ground=mix(forest,peak,np.clip(ridge*.73+e*.27+fine*.18,0,1))
 surface=mix(np.array(ocean)[None,None,:]*(.9+.25*fine[...,None]),ground,land)
 slope=np.gradient(e+ridge*.18,axis=1);surface*=np.clip(1+slope[...,None]*7,.63,1.32)
 curls=p+np.array([fb(p,3,4),fb(p,3.1,4),fb(p,3.3,4)])*.22
 storm=fb(curls*np.array([1,1.65,1])[:,None,None],4.2,7)
 clouds=np.clip((storm-.04)*.95,0,.90)
 tex=np.uint8(mix(surface,cloud,clouds).clip(0,255))
 Image.fromarray(tex).save(A/f'world{k}.jpg',quality=92)
 city=(np.exp(-np.abs(e-.085)*8)*land*(fine>.36)*(.4+.6*rng.random(e.shape))*(1-clouds))
 glow=np.uint8(city[...,None]*np.array([255,197,104]));Image.fromarray(glow).save(A/f'lights{k}.jpg',quality=83)
 # Atmospheric portrait also serves as an interactive fallback surface.
 s=840; yy,xx=np.mgrid[:s,:int(s*1.65)];x=(xx-xx.shape[1]/2)/(s*.348);y=(s/2-yy)/(s*.348)
 r=x*x+y*y;z=np.sqrt(np.maximum(0,1-r)); phi=np.arctan2(z,-x)+.4
 u=(phi/(2*np.pi)*W).astype(int)%W;v=np.int32(np.arccos(y.clip(-1,1))/np.pi*(H-1))
 light=np.maximum(0,-.55*x+.72*y+.31*z);c=tex[v,u]*(.15+1.15*light[...,None])+glow[v,u]*(1-light[...,None])*.85
 rim=(1-z)**7*(.28+light*.6);tint=np.array([[121,206,185],[147,190,237],[244,180,138]][k]);c+=rim[...,None]*tint
 alpha=np.where(r<=1,255,np.exp(-np.maximum(np.sqrt(r)-1,0)*60)*80)
 c=np.where((r<=1)[...,None],c,tint)
 # Elliptical particle belt; screen-space depth distinguishes front and back.
 angle=.32+k*.09; X=x*np.cos(angle)+y*np.sin(angle);Y=-x*np.sin(angle)+y*np.cos(angle)
 rad=np.sqrt(X**2+(Y/.23)**2);stripe=(.4+.6*np.sin(rad*214)**2)*(.5+.5*np.sin(rad*49+.3)**2)
 mask=(rad>1.31)&(rad<1.95)&((Y<0)|(r>1))
 dust=np.clip((.27+stripe*.7)*rng.random(r.shape)*mask,0,1)
 ring=np.array([[181,197,164],[169,195,218],[227,191,142]][k])
 a=alpha/255;out=dust+a*(1-dust);c=(dust[...,None]*ring+c*(a*(1-dust))[...,None])/np.maximum(.00001,out[...,None])
 im=Image.fromarray(np.uint8(np.dstack([c.clip(0,255),out*255])));im.save(A/f'portrait{k}.webp',quality=87)
