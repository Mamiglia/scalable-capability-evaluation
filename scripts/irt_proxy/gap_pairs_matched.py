import numpy as np, pandas as pd, matplotlib.pyplot as plt
G="judge_results/glm_correction"
SCRATCH="/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/65e871a7-3123-4881-8df7-bceb26a9b7c8/scratchpad"
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":6,"xtick.labelsize":7,"ytick.labelsize":7,
 "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,
 "font.family":"sans-serif","font.sans-serif":["Arial"],"svg.fonttype":"none"})
GREY,BLUE="#7f7f7f","#2a78d6"

d=pd.read_csv(f"{SCRATCH}/expansion_gpqa_glm.csv",usecols=["judge","left_model","right_model","chosen_left","left_capability","right_capability"])
d=d[d.left_capability!=d.right_capability].copy()
d["gap"]=(d.left_capability-d.right_capability).abs()
d["correct"]=np.where(d.left_capability>d.right_capability,d.chosen_left,1-d.chosen_left)
d["pair"]=["|".join(sorted(p)) for p in zip(d.left_model,d.right_model)]
edges=np.arange(0,0.75,0.1); d["bin"]=pd.cut(d.gap,edges)

gp=pd.read_csv(f"{G}/tables/gap_pairs.csv"); gp=gp[gp.benchmark.eq("gpqa")&gp.irt_model.eq("fixed_3pl")&gp.method.eq("corrected")].copy()
gp["pair"]=["|".join(sorted(p)) for p in zip(gp.model_left,gp.model_right)]
gp=gp[gp.pair.isin(set(d.pair))].copy(); gp["bin"]=pd.cut(gp.absolute_true_gap,edges)
print("judges:",d.judge.nunique(),"| judged pairs:",d.pair.nunique(),"| corrected matched:",gp.pair.nunique())

fig,ax=plt.subplots(figsize=(3.5,2.6))
ax.axhline(0.5,ls="--",lw=0.8,c="grey",zorder=0)
pp=d.groupby("pair").agg(gap=("gap","first"),acc=("correct","mean")).reset_index()
ax.plot(pp.gap,pp.acc,"o",ms=2.3,mec="none",c="#cccccc",alpha=0.55,zorder=1)          # individual pairs
for j,g in d.groupby("judge"):                                                          # individual judges
    b=g.groupby("bin",observed=True).correct.mean().dropna()
    ax.plot([iv.mid for iv in b.index],b.to_numpy(),lw=0.6,c=GREY,alpha=0.5,zorder=2)
pj=d.groupby(["bin","judge"],observed=True).correct.mean()
rm=pj.groupby(level="bin",observed=True).mean().dropna()
ax.plot([iv.mid for iv in rm.index],rm.to_numpy(),lw=1.5,c="black",zorder=4)             # raw mean
cb=gp.groupby("bin",observed=True).ordering_correct.mean().dropna()
ax.plot([iv.mid for iv in cb.index],cb.to_numpy(),lw=1.5,c=BLUE,zorder=5)                # corrected
ax.set_xlabel("True capability gap"); ax.set_ylabel("Ordering accuracy")
ax.set_xticks([0,0.3,0.6]); ax.set_yticks([0,0.5,1]); ax.set_xlim(-0.02,0.66); ax.set_ylim(-0.02,1.02)
for s,(lo,hi) in (("bottom",(0,0.6)),("left",(0,1))): ax.spines[s].set_position(("outward",3)); ax.spines[s].set_bounds(lo,hi)
h=[plt.Line2D([],[],marker="o",ls="",ms=3,mec="none",c="#cccccc",label="individual pairs"),
   plt.Line2D([],[],lw=0.6,c=GREY,alpha=0.7,label="individual judges"),
   plt.Line2D([],[],lw=1.5,c="black",label="raw judges (mean)"),
   plt.Line2D([],[],lw=1.5,c=BLUE,label="corrected ordering")]
ax.legend(handles=h,frameon=False,loc="lower right",fontsize=6,handlelength=1.4,borderaxespad=0.2)
fig.tight_layout()
for suf in ("png","svg"): fig.savefig(f"{G}/gap_pairs_raw_vs_corrected.{suf}",dpi=300,bbox_inches="tight")
print("done")
