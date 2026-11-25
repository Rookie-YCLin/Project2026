function phase03_figures(runDir, phase02Dir, ids)
% 由 MATLAB 區塊統計、獨立矩陣測試及盲式 BER 產出第三階段圖表。
arguments
    runDir (1,1) string
    phase02Dir (1,1) string
    ids (1,:) double = 1:24
end
stats=fullfile(runDir,'05_統計分析');
figDir=fullfile(runDir,'06_MATLAB圖表');
if ~isfolder(figDir), mkdir(figDir); end
vectors=["U1","U2","U3","U4","V1","V2","V3","V4"];
methods=["S","G","R","P"];
palette=[.03 .35 .62;.78 .27 .04;.07 .5 .34;.55 .25 .62];
styles=["-","--","-.",":"];
schema=["ber_psnr","matrix_matlab","decision_matlab","relation_matlab"];
tables=cell(32,numel(schema));
v=0;
for m=methods
    for vec=vectors
        v=v+1;
        for j=1:numel(schema)
            tables{v,j}=readtable(fullfile(stats,m+vec+"_"+schema(j)+".csv"),'TextType','string');
        end
    end
end
BER=vertcat(tables{:,1});
B=BER(~isnan(BER.qf),:);
B.ber_pct=100*B.ber; B.block_pct=100*B.block_ber;
A=vertcat(tables{:,2}); A=A(A.scope=="All",:);
E=A(A.comparison=="embedding",:);
Q=vertcat(tables{:,3}); Q=Q(Q.scope=="All",:);
R=vertcat(tables{:,4}); R=R(R.scope=="All",:);
paired=readtable(fullfile(stats,'same_target_loss_matlab.csv'),'TextType','string');
reductions=readtable(fullfile(stats,'same_target_reduction_matlab.csv'),'TextType','string');
titles=["Final 512-bit BER","PSNR relative to original","Frozen angular step and PNG PSNR", ...
    "PNG block BER","Theoretical embedding loss","Theory, reconstruction and realized loss", ...
    "Common target angles at frozen P settings","P distortion reduction vs. R and G", ...
    "P primal–dual gap","P solver time","Orthogonality residual by method","Singular value displacement in B", ...
    "B direction after a fresh SVD","G center reselection and failure","PNG identifiability by spectral class", ...
    "JPEG-only squared Frobenius perturbation","JPEG-only left projective rotation", ...
    "JPEG-only right projective rotation","Normalized absolute QIM offset","Margin and block correctness", ...
    "Nearest singular gap and angular drift","Block and payload BER","QF 1–30 final BER", ...
    "Phase 02 vs. Phase 03 matched-PSNR BER"];
for id=ids
    f=figure('Visible','off','Color','white','Units','inches','Position',[1 1 10 5.2], ...
        'DefaultAxesFontName','Times New Roman','DefaultAxesFontSize',9,'DefaultTextFontName','Times New Roman');
    done=onCleanup(@()close(f));
    tl=tiledlayout(f,2,4,'TileSpacing','compact','Padding','compact');
    title(tl,titles(id),'FontName','Times New Roman','FontWeight','bold');
    evidence=cell(8,1);
    for k=1:numel(vectors)
        ax=nexttile(tl);
        vec=vectors(k);
        labels=methods+vec;
        tA=A(ismember(A.method,labels),:);
        tB=B(ismember(B.method,labels),:);
        tE=E(ismember(E.method,labels),:);
        tQ=Q(ismember(Q.method,labels),:);
        tR=R(ismember(R.method,labels),:);
        cals=BER(isnan(BER.qf)&ismember(BER.method,labels),:);
        title(ax,vec);
        hold(ax,'on'); grid(ax,'on'); box(ax,'on');
        switch id
            case {1,2,23}
                x=tB;
                field="ber_pct"; y="Final BER (%)";
                if id==2, field="psnr_original_db"; y="PSNR (dB)"; end
                if id==23, x=x(x.qf<=30,:); end
                curves(ax,x,field,labels,palette,styles,"JPEG quality factor",y);
            case 3
                x=cals;
                yyaxis(ax,'left');
                bars(ax,x,"q_rad",labels,palette,"Step (rad)");
                yyaxis(ax,'right');
                plot(ax,methodIndex(x.method,labels),x.psnr_original_db,'kd','MarkerFaceColor','w');
                yline(ax,39.9,':'); yline(ax,40.1,':'); ylabel(ax,'PSNR (dB)');
            case 4
                x=cals; x.value=100*x.block_ber;
                bars(ax,x,"value",labels,palette,"Block BER (%)");
            case {5,6}
                x=tA(tA.qf==0 & tA.metric=="D_F",:);
                if id==5
                    x=x(x.comparison=="A_to_B",:);
                    bars(ax,x,"mean",labels,palette,"Mean squared Frobenius norm");
                else
                    keyBars=grouped(ax,x,labels,"comparison",["A_to_B","B_to_B0","A_to_B0"],"Mean squared Frobenius norm");
                end
            case 7
                x=paired(paired.vector==vec & paired.scope=="G_feasible",:);
                for j=1:4
                    v=x(x.method==methods(j),:);
                    errorbar(ax,j,v.median,v.median-v.p10,v.p90-v.median,'o','Color',palette(j,:), ...
                        'MarkerFaceColor',palette(j,:),'MarkerSize',3.8,'LineWidth',1);
                end
                xlim(ax,[.5 4.5]); xticks(ax,1:4); xticklabels(ax,methods);
                ylabel(ax,'Normalized squared loss');
            case 8
                x=reductions(reductions.vector==vec,:);
                for j=2:3
                    v=x(x.baseline==methods(j),:);
                    errorbar(ax,j-1,v.median,v.median-v.p10,v.p90-v.median,'o','Color',palette(j,:),...
                        'MarkerFaceColor',palette(j,:),'MarkerSize',3.8,'LineWidth',1);
                end
                xlim(ax,[.5 2.5]); xticks(ax,1:2); xticklabels(ax,["vs. G","vs. R"]);
                ylabel(ax,'Loss reduction (%)');
            case {9,10,13}
                keys=["certificate_gap","solve_seconds_per_block_batch_average","B_angle_target_error_success"];
                idx=find([9,10,13]==id);
                y=["Scaled trace gap","Time per block (s)","Mean direction error (rad)"];
                x=tE(tE.metric==keys(idx),:);
                bars(ax,x,"mean",labels,palette,y(idx));
            case 11
                x=tE(tE.metric=="orthogonality",:);
                for j=1:4
                    row=x(x.method==labels(j),:);
                    fill=palette(j,:);
                    if row.classification=="exploratory", fill='none'; end
                    plot(ax,j,row.mean,'o','Color',palette(j,:),'MarkerFaceColor',fill,'MarkerSize',5);
                end
                ax.YScale='log';
                ylim(ax,[1e-18 1]);
                yticks(ax,10.^(-18:6:0));
                xlim(ax,[.5 4.5]); xticks(ax,1:4); xticklabels(ax,methods);
                ylabel(ax,'Mean orthogonality residual');
            case 12
                x=tA(tA.qf==0 & tA.comparison=="A_to_B" & startsWith(tA.metric,"delta_sigma_"),:);
                keyBars=grouped(ax,x,labels,"metric","delta_sigma_"+(1:4),"Mean signed singular value change");
            case 14
                x=tE(startsWith(tE.method,"G") & ismember(tE.metric,["reselected","embed_failure"]),:);
                x.percent=100*x.mean;
                x.event=replace(replace(x.metric,"reselected","Reselected"),"embed_failure","Failed");
                bar(ax,categorical(x.event),x.percent,'FaceColor',palette(2,:));
                ylabel(ax,'Blocks (%)');
            case 15
                x=tR(tR.qf==0 & tR.rank_class>0,:);
                x.value=100*(1-x.block_ber);
                for j=1:4
                    y=x(x.method==labels(j) & x.count>0,:);
                    plot(ax,y.rank_class,y.value,'o','Color',palette(j,:),'MarkerSize',4);
                end
                xticks(ax,1:5); xlim(ax,[.5 5.5]); ylim(ax,[0 100]);
                xlabel(ax,'Spectral class'); ylabel(ax,'Correct blocks (%)');
            case {16,17,18}
                key="D_F";
                if id==17, key="theta_U_"+extractAfter(vec,1); end
                if id==18, key="theta_V_"+extractAfter(vec,1); end
                x=tA(tA.qf>0 & tA.comparison=="B0_to_C" & tA.metric==key,:);
                label="Mean squared Frobenius norm";
                if id>16, label="Mean angle (rad)"; end
                curves(ax,x,"mean",labels,palette,styles,"JPEG quality factor",label);
            case 19
                x=tQ(tQ.qf>0 & tQ.metric=="abs_offset",:);
                x.abs_norm=x.mean./cals.q_rad(methodIndex(x.method,cals.method.'));
                curves(ax,x,"abs_norm",labels,palette,styles,"JPEG quality factor","Mean |offset| / q");
            case 20
                x=tQ(tQ.qf>0 & ismember(tQ.metric,["target_margin","block_error"]),:);
                yyaxis(ax,'left');
                curves(ax,x(x.metric=="target_margin",:),"mean",labels,palette,styles,"JPEG quality factor","Margin (rad)");
                yyaxis(ax,'right');
                x.correct_pct=100*(1-x.mean);
                curves(ax,x(x.metric=="block_error",:),"correct_pct",labels,palette,styles,"JPEG quality factor","Correct blocks (%)");
            case 21
                x=tR(tR.qf>0 & tR.rank_class==0,:);
                curves(ax,x,"pearson_gap_angle",labels,palette,styles,"JPEG quality factor","Pearson correlation");
                yline(ax,0,':');
            case 22
                x=tB;
                for j=1:4
                    y=x(x.method==labels(j),:);
                    scatter(ax,y.block_pct,y.ber_pct,7,palette(j,:),'filled');
                end
                xlabel(ax,'Block BER (%)'); ylabel(ax,'Final BER (%)');
            case 24
                prior=readtable(fullfile(phase02Dir,'07_統計分析','all_methods_ber_psnr.csv'),'TextType','string');
                prior=prior(prior.condition~="unattacked",["method","qf","ber_pct"]);
                prior.classification=repmat("phase02",height(prior),1);
                x=tB(tB.classification=="formal",["method","qf","ber_pct","classification"]);
                for p=unique(prior.method).'
                    s=prior(prior.method==p,:);
                    shade=[.77 .77 .77]; width=.55; lineStyle="-";
                    if p=="G0", shade=[.36 .36 .36]; lineStyle="--"; width=1; end
                    if p=="S1", shade=[.12 .12 .12]; width=1; end
                    plot(ax,s.qf,s.ber_pct,'Color',shade,'LineWidth',width,'LineStyle',lineStyle);
                end
                curves(ax,x,"ber_pct",labels,palette,styles,"JPEG quality factor","Final BER (%)");
                x=[prior;x];
        end
        x.panel=repmat(vec,height(x),1);
        evidence{k}=x;
    end
    details="S blue, G orange, R green, P purple. Hollow curve markers or bars indicate exploratory calibration.";
    switch id
        case 3
            details="Bars show step size, diamonds show PNG PSNR. Outlined bars indicate exploratory calibration.";
        case 6
            legendLabels=["A → B","B → B₀","A → B₀"];
        case 11
            details="The vertical axis is logarithmic. Hollow markers indicate exploratory calibration.";
        case 12
            legendLabels=["Δσ₁","Δσ₂","Δσ₃","Δσ₄"];
        case 7
            details="All four methods use the same feasible blocks. Points show medians and bars span P10 to P90.";
        case 8
            details="Median and P10 to P90. Baseline loss exceeds 10^{-12} and 100 times the certificate width.";
        case 14
            details="Reselection and failure are measured over all 38,088 blocks for each G method.";
        case {15,22}
            details="S blue, G orange, R green, P purple. Point clouds include every calibration status.";
        case 17
            details="Panels name the embedded vector. Drift is measured on U. S blue, G orange, R green, P purple.";
        case 18
            details="Panels name the embedded vector. Drift is measured on V. S blue, G orange, R green, P purple.";
        case 24
            details="Phase 02 uses black for S1 and dashed grey for G0. Coloured curves show Phase 03 formal groups.";
    end
    if ismember(id,[6 12])
        key=legend(keyBars,legendLabels,'Orientation','horizontal','FontName','Times New Roman','FontSize',9);
        key.Layout.Tile='south';
    else
        xlabel(tl,details,'FontName','Times New Roman','FontSize',8);
    end
    path=fullfile(figDir,compose("Fig%02d",id));
    writetable(vertcat(evidence{:}),path+".csv");
    savefig(f,path+".fig");
    exportgraphics(f,path+".png",'Resolution',300,'BackgroundColor','white');
    try
        exportgraphics(f,path+".svg",'ContentType','vector','BackgroundColor','white');
    catch
        print(f,path+".svg",'-dsvg','-painters');
    end
    exportgraphics(f,path+".pdf",'ContentType','vector','BackgroundColor','white');
    fprintf('Exported %s\n',path);
    clear done
end
end

function idx=methodIndex(method,labels)
[~,idx]=ismember(method,labels);
assert(all(idx>0));
end

function curves(ax,x,field,labels,palette,styles,xname,yname)
for j=1:numel(labels)
    t=sortrows(x(x.method==labels(j),:),"qf");
    if isempty(t), continue; end
    plot(ax,t.qf,t.(field),'Color',palette(j,:),'LineStyle',styles(j),'LineWidth',1.2);
    if t.classification(1)=="exploratory"
        k=1:10:height(t); plot(ax,t.qf(k),t{k,field},'o','Color',palette(j,:),'MarkerSize',2.5);
    end
end
xlabel(ax,xname); ylabel(ax,yname);
end

function bars(ax,x,field,labels,palette,label)
for j=1:numel(labels)
    row=x(x.method==labels(j),:);
    if isempty(row), continue; end
    c=palette(j,:);
    if row.classification=="exploratory"
        bar(ax,j,row.(field),'FaceColor','none','EdgeColor',c,'LineWidth',1.2);
    else
        bar(ax,j,row.(field),'FaceColor',c,'EdgeColor','none');
    end
end
xticks(ax,1:4); xticklabels(ax,extractBetween(labels,1,1)); xlim(ax,[.45 4.55]); ylabel(ax,label);
end

function handles=grouped(ax,x,labels,key,groups,label)
[found,idx]=ismember(x.(key),groups);
assert(all(found));
rows=methodIndex(x.method,labels);
v=nan(numel(labels),numel(groups));
v(sub2ind(size(v),rows,idx))=x.mean;
handles=bar(ax,v);
xticks(ax,1:4); xticklabels(ax,extractBetween(labels,1,1)); ylabel(ax,label);
end
