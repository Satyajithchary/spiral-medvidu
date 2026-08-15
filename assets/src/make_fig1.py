import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np

plt.rcParams.update({"font.family":"serif",
    "font.serif":["Nimbus Roman","Times New Roman","DejaVu Serif"],
    "pdf.fonttype":42,"ps.fonttype":42,"text.usetex":False})

C = {"data":("#2E7D64","#E9F4F0"), "train":("#5B52B5","#EFEEFB"),
     "inf":("#B85C36","#FBEFEA"), "dec":("#9A7B24","#FBF4E2"),
     "cal":("#2F6E9E","#E9F1F8")}
GE, GY, DK = "#8C8A82", "#61605B", "#1A1A18"

W,H = 7.0, 2.62
fig = plt.figure(figsize=(W,H)); ax = fig.add_axes([0,0,1,1])
YM = 100*H/W; ax.set_xlim(0,100); ax.set_ylim(0,YM); ax.axis("off")

def rb(x,y,w,h,fc="white",ec=GE,lw=0.6,r=0.6,z=2,ls="-",a=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc,ec=ec,lw=lw,zorder=z,linestyle=ls,alpha=a))
def tx(x,y,s,fs=5.2,c=DK,ha="center",va="center",w="normal",st="normal",z=6):
    ax.text(x,y,s,fontsize=fs,color=c,ha=ha,va=va,weight=w,style=st,zorder=z)
def ar(x1,y1,x2,y2,ec=GE,lw=0.6,ls="-",ms=5,z=7):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=ms,
                 lw=lw,color=ec,linestyle=ls,shrinkA=0,shrinkB=0,zorder=z))
def elbow(p,ec=GE,lw=0.6,ls="-"):
    ax.add_patch(PathPatch(Path(p,[Path.MOVETO]+[Path.LINETO]*(len(p)-1)),
                 fc="none",ec=ec,lw=lw,ls=ls,zorder=7))
    ar(p[-2][0],p[-2][1],p[-1][0],p[-1][1],ec=ec,lw=lw,ls=ls)

Y0, PH = 2.4, YM-4.2
XS = [0.8, 20.4, 40.0, 61.2, 79.0]
WS = [18.8, 18.8, 20.4, 17.0, 20.2]
TITLES = [("1. Data Enrichment","data"),("2. Training","train"),
          ("3. Self-Refining Inference","inf"),("4. Ordinal Decoding","dec"),
          ("5. Output Calibration","cal")]
for (x,w,(t,k)) in zip(XS,WS,TITLES):
    ec,fc = C[k]
    rb(x,Y0,w,PH,fc=fc,ec=ec,lw=0.85,r=1.2,z=1)
    tx(x+w/2,Y0+PH-2.2,t,5.9,ec,w="bold")

# ---------- 1 data ----------
x,w = XS[0],WS[0]; ec=C["data"][0]
rb(x+1.4,Y0+PH-9.4,w-2.8,5.6,ec=ec)
tx(x+w/2,Y0+PH-5.6,"heterogeneous sources",5.0,DK)
tx(x+w/2,Y0+PH-7.6,"8 corpora, 6,270 rows",4.5,GY)
CH=["instrument","action","target","span","region","phase","vocab"]
cw,ch,g=5.1,2.5,0.7
bus=Y0+PH-11.6
ax.plot([x+2.6,x+w-2.6],[bus,bus],color=ec,lw=0.5,zorder=4)
ar(x+w/2,Y0+PH-9.6,x+w/2,bus+0.05,ec=ec,lw=0.5,ms=3.5)
for row,items in enumerate([CH[:3],CH[3:6],CH[6:]]):
    rw=len(items)*cw+(len(items)-1)*g; x0=x+(w-rw)/2; yy=Y0+PH-15.2-row*(ch+1.0)
    for i,c in enumerate(items):
        xx=x0+i*(cw+g); rb(xx,yy,cw,ch,ec=ec,r=0.35); tx(xx+cw/2,yy+ch/2,c,4.2,ec)
        ar(xx+cw/2,bus,xx+cw/2,yy+ch+0.12,ec=ec,lw=0.4,ms=3)
tx(x+w/2,Y0+7.6,"7 typed streams $\\rightarrow$ auxiliary QA",4.6,ec,st="italic")
bx,bw=x+3.0,w-6.0
ax.add_patch(Rectangle((bx,Y0+3.4),bw*6270/8150,1.6,fc=GE,ec="none",zorder=3))
ax.add_patch(Rectangle((bx,Y0+5.4),bw,1.6,fc=ec,ec="none",zorder=3))
tx(bx-0.5,Y0+4.2,"6,270",4.2,GY,ha="right"); tx(bx-0.5,Y0+6.2,"8,150",4.2,ec,ha="right")

# ---------- 2 training ----------
x,w = XS[1],WS[1]; ec=C["train"][0]
tx(x+w/2,Y0+PH-5.0,"gradient share per (dataset, task)",4.6,GY,st="italic")
sh=np.array([1395,1391,1382,725,600,394,233,150],float); sh/=sh.sum()
bl=sh**0.5; bl/=bl.sum()
sw,sg=1.28,0.42; x0=x+(w-(8*sw+7*sg))/2; by=Y0+PH-14.2
for i in range(8):
    xx=x0+i*(sw+sg)
    ax.add_patch(Rectangle((xx,by),sw,sh[i]*19,fc=GE,ec="none",zorder=3))
    ax.add_patch(Rectangle((xx,by),sw,bl[i]*19,fc="none",ec=ec,lw=0.6,
                 ls=(0,(1.3,1.0)),zorder=4))
tx(x+w/2,Y0+PH-15.9,"uniform vs $\\tau{=}0.5$",4.5,ec)
ar(x+w/2,Y0+PH-16.9,x+w/2,Y0+PH-18.4,ec=ec,lw=0.6,ms=4)

MY, MH = Y0+4.4, PH-23.6
rb(x+1.5,MY,w-3.0,MH,ec=ec,r=0.7)
tx(x+w/2,MY+MH-1.9,"Qwen3-VL-4B",5.2,DK,w="bold")
inner_top = MY+MH-4.2
rb(x+2.8,MY+1.6,5.4,inner_top-MY-1.6,ec=GE,r=0.3,ls=(0,(1.4,1.1)))
tx(x+5.5,MY+1.6+(inner_top-MY-1.6)*0.62,"ViT",4.4,GY)
tx(x+5.5,MY+1.6+(inner_top-MY-1.6)*0.28,"frozen",4.0,GY,st="italic")
bh=(inner_top-MY-1.6-1.2)/3
for k in range(3):
    yy=MY+1.6+k*(bh+0.6)
    rb(x+9.4,yy,4.6,bh,ec=ec,r=0.3)
    tx(x+11.7,yy+bh/2,"block",4.0,DK)
    ax.add_patch(Rectangle((x+14.4,yy+bh*0.22),0.95,bh*0.56,fc=ec,ec="none",zorder=4))
tx(x+11.7,inner_top+1.0,"LoRA adapters",4.2,ec)
tx(x+w/2,Y0+1.7,"corrupted-prior conditioning",4.5,ec,st="italic")

# ---------- 3 inference ----------
x,w = XS[2],WS[2]; ec=C["inf"][0]
ty=Y0+PH-8.6
tx(x+1.6,Y0+PH-4.8,"temporal zoom",4.7,ec,ha="left",st="italic")
ax.add_patch(Rectangle((x+1.8,ty+1.9),w-3.6,1.4,fc="white",ec=GE,lw=0.5,zorder=3))
for i in range(11):
    xx=x+1.8+i*(w-3.6)/10; ax.plot([xx,xx],[ty+1.9,ty+2.3],color=GE,lw=0.4,zorder=4)
ax.add_patch(Rectangle((x+7.2,ty+1.9),5.4,1.4,fc=GE,ec="none",alpha=0.45,zorder=3))
elbow([(x+6.2,ty+1.8),(x+6.2,ty+1.0),(x+13.6,ty+1.0),(x+13.6,ty+1.8)],ec=ec,lw=0.5)
ar(x+9.9,ty+0.95,x+9.9,ty-0.35,ec=ec,lw=0.5,ms=4)
ax.add_patch(Rectangle((x+1.8,ty-2.4),w-3.6,1.4,fc="white",ec=ec,lw=0.5,zorder=3))
for i in range(21):
    xx=x+1.8+i*(w-3.6)/20; ax.plot([xx,xx],[ty-2.4,ty-2.0],color=ec,lw=0.3,zorder=4)
ax.add_patch(Rectangle((x+8.6,ty-2.4),3.6,1.4,fc=ec,ec="none",alpha=0.5,zorder=3))
tx(x+w/2,ty-3.6,"resample inside hypothesis",4.3,ec)
sy2=Y0+3.2
tx(x+1.6,sy2+8.6,"spatial zoom",4.7,ec,ha="left",st="italic")
ax.add_patch(Rectangle((x+2.2,sy2+0.8),7.4,6.6,fc="white",ec=GE,lw=0.5,zorder=3))
ax.add_patch(Rectangle((x+5.2,sy2+2.8),2.6,2.2,fc="none",ec=GE,lw=0.6,
             ls=(0,(1.4,1.0)),zorder=4))
tx(x+5.9,sy2-0.8,"coarse box",4.2,GY)
ar(x+10.1,sy2+4.1,x+11.6,sy2+4.1,ec=ec,lw=0.6,ms=4)
ax.add_patch(Rectangle((x+12.0,sy2+0.8),7.4,6.6,fc="white",ec=ec,lw=0.6,zorder=3))
ax.add_patch(Rectangle((x+14.4,sy2+2.2),3.2,3.0,fc="none",ec=ec,lw=0.8,zorder=4))
tx(x+15.7,sy2-0.8,"crop, re-predict",4.2,ec)

# ---------- 4 decoding ----------
x,w = XS[3],WS[3]; ec=C["dec"][0]
tx(x+w/2,Y0+PH-5.0,"$K{=}5$ sampled generations",4.6,GY,st="italic")
for i,v in enumerate([0,1,1,0,2]):
    xx=x+3.0+i*2.8
    ax.add_patch(Circle((xx,Y0+PH-9.0),0.95,fc="white",ec=ec,lw=0.6,zorder=4))
    tx(xx,Y0+PH-9.0,str(v),4.6,DK)
ar(x+w/2,Y0+PH-10.6,x+w/2,Y0+PH-12.4,ec=ec,lw=0.6,ms=4)
rb(x+2.0,Y0+PH-17.6,w-4.0,4.8,ec=ec,r=0.5)
tx(x+w/2,Y0+PH-14.3,"$\\hat v=\\mathrm{round}(\\mathbb{E}[v])$",5.0,DK)
tx(x+w/2,Y0+PH-16.4,"$0.8 \\rightarrow 1$",4.5,ec)
rb(x+2.0,Y0+3.0,w-4.0,4.4,ec=GE,r=0.5,ls=(0,(1.4,1.1)))
tx(x+w/2,Y0+5.9,"greedy",4.6,GY)
tx(x+w/2,Y0+3.9,"returns 0",4.2,GY,st="italic")
tx(x+w/2,Y0+8.6,"recovers intermediate scores",4.3,ec,st="italic")

# ---------- 5 calibration ----------
x,w = XS[4],WS[4]; ec=C["cal"][0]
items=[("budget allocation","from reference length"),
       ("question-aware keying","(id, task, question)"),
       ("span and box completion","clamp, replicate"),
       ("terminology normalisation","ontology lookup")]
for i,(a_,b_) in enumerate(items):
    yy=Y0+PH-9.6-i*4.6
    rb(x+1.5,yy,w-3.0,3.8,ec=ec,r=0.4)
    tx(x+w/2,yy+2.45,a_,4.5,DK); tx(x+w/2,yy+1.05,b_,4.0,GY)
rb(x+1.5,Y0+2.4,w-3.0,4.0,fc=ec,ec=ec,r=0.5)
tx(x+w/2,Y0+4.4,"benchmark-compliant output",4.5,"white",w="bold")

for i in range(4):
    ar(XS[i]+WS[i]+0.15,Y0+PH/2,XS[i+1]-0.15,Y0+PH/2,ec=GY,lw=0.8,ms=6)

fig.savefig("/mnt/user-data/outputs/spiral-medvidu/assets/fig1_spiral_architecture.pdf",bbox_inches="tight",pad_inches=0.06,facecolor="white")
fig.savefig("/mnt/user-data/outputs/spiral-medvidu/assets/fig1_spiral_architecture.png",dpi=200,bbox_inches="tight",pad_inches=0.06,facecolor="white")
print("ok")
