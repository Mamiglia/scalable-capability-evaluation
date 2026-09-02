import numpy as np, pandas as pd, matplotlib.pyplot as plt
G="judge_results/glm_correction"
SCRATCH="/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/65e871a7-3123-4881-8df7-bceb26a9b7c8/scratchpad"
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":6,"xtick.labelsize":7,"ytick.labelsize":7,
 "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,
 "font.family":"sans-serif","font.sans-serif":["Arial"],"svg.fonttype":"none"})
PAL=["#2a78d6","#eb6834","#1baf7a","#d62728","#8551b5","#e6a000","#e87ba4","#17a2b8","#8c564b","#444444","#7f9a00","#00a0b0"]

# true + corrected capability per model, from gap_pairs
gp=pd.read_csv(f"{G}/tables/gap_pairs.csv"); gp=gp[gp.benchmark.eq("gpqa")&gp.irt_model.eq("fixed_3pl")&gp.method.eq("corrected")]
true=pd.concat([gp[["model_left","observed_left"]].set_axis(["m","v"],axis=1),gp[["model_right","observed_right"]].set_axis(["m","v"],axis=1)]).drop_duplicates("m").set_index("m").v
corr=pd.concat([gp[["model_left","prediction_left"]].set_axis(["m","v"],axis=1),gp[["model_right","prediction_right"]].set_axis(["m","v"],axis=1)]).drop_duplicates("m").set_index("m").v

d=pd.read_csv(f"{SCRATCH}/expansion_gpqa_glm.csv",usecols=["judge","left_model","right_model","chosen_left"])
for side in ("left","right"):
    d[f"{side}_t"]=d[f"{side}_model"].map(true); d[f"{side}_c"]=d[f"{side}_model"].map(corr)
d=d.dropna(subset=["left_t","right_t","left_c","right_c"])
def curve(sub,tcol_l,tcol_r,edges):
    s=sub[sub[tcol_l]!=sub[tcol_r]].copy()
    s["gap"]=(s[tcol_l]-s[tcol_r]).abs()
    s["correct"]=np.where(s[tcol_l]>s[tcol_r],s.chosen_left,1-s.chosen_left)
    b=s.groupby(pd.cut(s.gap,edges),observed=True).correct.mean().dropna()
    return [iv.mid for iv in b.index],b.to_numpy()
edges=np.arange(0,0.75,0.1)
judges=sorted(d.judge.unique()); colour=dict(zip(judges,PAL))
fig,ax=plt.subplots(figsize=(4.2,3.0))
ax.axhline(0.5,ls="--",lw=0.8,c="grey",zorder=0)
for j in judges:
    sub=d[d.judge.eq(j)]
    x0,y0=curve(sub,"left_t","right_t",edges)
    x1,y1=curve(sub,"left_c","right_c",edges)
    ax.plot(x0,y0,ls=":",lw=0.9,c=colour[j],alpha=0.7,zorder=2)
    ax.plot(x1,y1,ls="-",lw=1.1,c=colour[j],zorder=3)
ax.set_xlabel("Capability gap"); ax.set_ylabel("Judge accuracy")
ax.set_xticks([0,0.3,0.6]); ax.set_yticks([0,0.5,1]); ax.set_xlim(-0.02,0.66); ax.set_ylim(-0.02,1.02)
for s,(lo,hi) in (("bottom",(0,0.6)),("left",(0,1))): ax.spines[s].set_position(("outward",3)); ax.spines[s].set_bounds(lo,hi)
jh=[plt.Line2D([],[],lw=1.4,c=colour[j],label=j) for j in judges]
sh=[plt.Line2D([],[],ls=":",lw=1,c="dimgrey",label="original (vs true)"),
    plt.Line2D([],[],ls="-",lw=1.2,c="dimgrey",label="corrected (vs corrected cap.)")]
leg1=ax.legend(handles=jh,frameon=False,ncol=2,fontsize=5.5,loc="lower right",bbox_to_anchor=(1.0,0.0),handlelength=1.3)
ax.add_artist(leg1)
ax.legend(handles=sh,frameon=False,fontsize=6,loc="upper left",bbox_to_anchor=(0.0,1.0),handlelength=2)
fig.tight_layout()
for suf in ("png","svg"): fig.savefig(f"{G}/gap_curve_judge_orig_vs_corrected.{suf}",dpi=300,bbox_inches="tight")
print("judges",len(judges),"| written gap_curve_judge_orig_vs_corrected")
