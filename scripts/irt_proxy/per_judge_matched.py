import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import kendalltau
G="judge_results/glm_correction"
plt.rcParams.update({"font.size":7,"axes.labelsize":7,"legend.fontsize":6,"xtick.labelsize":5.5,"ytick.labelsize":5.5,
 "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
 "font.family":"sans-serif","font.sans-serif":["Arial"],"svg.fonttype":"none"})
GREY,BLUE="#8a8a8a","#2a78d6"
lomo=pd.read_csv("irt_results/easy_item_prediction/tables/lomo_predictions.csv")
elo=lomo[lomo.irt_model.eq("fixed_3pl")&lomo.fit.eq("gpqa")&lomo.method.eq("elo")][["judge","model","prediction","observed"]].dropna()
corr=pd.read_csv(f"{G}/tables/predictions.csv"); corr=corr[corr.method.eq("corrected")].set_index("model").prediction
short=lambda s:s.replace("-a3b-instruct","").replace("-a4b-it","-a4b").replace("-0731","").replace("consensus","pooled")
judges=[j for j in sorted(elo.judge.unique()) if j!="consensus"]+["consensus"]
data={j:elo[elo.judge.eq(j)].assign(corr=lambda d:d.model.map(corr)).dropna(subset=["corr"]) for j in judges}
R=len(judges)
fig,axes=plt.subplots(R,2,figsize=(2.15,0.78*R+0.35))
for i,j in enumerate(judges):
    s=data[j]
    for col,(y,c) in enumerate([("prediction",GREY),("corr",BLUE)]):
        ax=axes[i,col]; ax.plot([0,1],[0,1],ls="--",lw=0.6,c="grey",zorder=0)
        ax.plot(s.observed,s[y],"o",ms=2.2,mec="none",c=c,alpha=0.9,zorder=2)
        for sd in ("bottom","left"): ax.spines[sd].set_position(("outward",2)); ax.spines[sd].set_bounds(0,1)
        ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xlim(-0.07,1.07); ax.set_ylim(-0.07,1.07); ax.set_box_aspect(1)
        ax.set_xticklabels(["0.0","1.0"] if i==R-1 else []); ax.set_yticklabels(["0.0","1.0"] if col==0 else [])
        ax.set_title(f"τ={kendalltau(s.observed,s[y]).statistic:.2f}",fontsize=5,color=c,pad=1.2)
    axes[i,0].text(-0.72,0.5,short(j),rotation=90,va="center",ha="center",transform=axes[i,0].transAxes,fontsize=5.5)
axes[0,0].annotate("raw",xy=(0.5,1.24),xycoords="axes fraction",ha="center",fontsize=6,color=GREY,annotation_clip=False)
axes[0,1].annotate("corrected",xy=(0.5,1.24),xycoords="axes fraction",ha="center",fontsize=6,color=BLUE,annotation_clip=False)
fig.supxlabel("Observed hard accuracy",fontsize=6.5,y=0.005); fig.supylabel("Predicted hard accuracy",fontsize=6.5,x=0.0)
fig.subplots_adjust(left=0.22,right=0.98,top=0.95,bottom=0.06,wspace=0.15,hspace=0.45)
for suf in ("png","svg"): fig.savefig(f"{G}/per_judge_predictions.{suf}",dpi=300,bbox_inches="tight")
import PIL.Image as I; print("size in:",[round(x/300,2) for x in I.open(f"{G}/per_judge_predictions.png").size])

# ---- Figure B: pre/post Kendall-tau slopegraph, line per judge + thick mean ----
PAL=["#2a78d6","#eb6834","#1baf7a","#d62728","#8551b5","#e6a000","#444444"]
rows=[(j,kendalltau(data[j].observed,data[j].prediction).statistic,kendalltau(data[j].observed,data[j]["corr"]).statistic) for j in judges]
fig,ax=plt.subplots(figsize=(2.3,2.5))
for (j,pre,post),c in zip(rows,PAL):
    ax.plot([0,1],[pre,post],"-o",lw=0.9,ms=3,mec="none",c=c,alpha=0.85,label=short(j))
mp,mq=np.mean([r[1] for r in rows]),np.mean([r[2] for r in rows])
ax.plot([0,1],[mp,mq],"-",lw=2.6,c="black",zorder=5)
ax.set_xticks([0,1]); ax.set_xticklabels(["raw\nElo","corrected"]); ax.set_xlim(-0.15,1.15)
ax.set_ylabel("Kendall τ (predicted vs observed)"); ax.set_ylim(0,1); ax.set_yticks([0,0.5,1])
for sd,(lo,hi) in (("bottom",(0,1)),("left",(0,1))): ax.spines[sd].set_position(("outward",3)); ax.spines[sd].set_bounds(lo,hi)
ax.legend(frameon=False,fontsize=5.5,loc="lower right",bbox_to_anchor=(1.03,-0.03),handlelength=1.2,handletextpad=0.3,labelspacing=0.25)
fig.tight_layout()
for suf in ("png","svg"): fig.savefig(f"{G}/tau_pre_post_by_judge.{suf}",dpi=300,bbox_inches="tight")
print("mean tau %.2f -> %.2f"%(mp,mq))
