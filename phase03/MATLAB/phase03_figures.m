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
M=readtable(fullfile(runDir,'01_數學驗證','stress_gate','matlab_cases.csv'),'TextType','string');
fromImage=M(startsWith(M.category,"platelet"),:);
fromImage.vector=fromImage.side+string(fromImage.j+1);
titles=["Final 512-bit BER","PSNR relative to original","Frozen angular step and PNG PSNR", ...
    "PNG block BER","Theoretical embedding loss","Theory, reconstruction and realized loss", ...
    "Common target angle on platelet blocks","P distortion reduction vs. R and G", ...
    "P primal–dual gap","P solver time","P orthogonality error","Singular value displacement in B", ...
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
                    grouped(ax,x,labels,"comparison",["A_to_B","B_to_B0","A_to_B0"],"Mean squared Frobenius norm");
                end
            case 7
                x=fromImage(fromImage.vector==vec,:);
                values=x{:,{'D_S','D_G','D_R','D_P'}};
                markers=["o","+","^","s"];
                for j=1:4
                    plot(ax,1:height(x),values(:,j),markers(j),'Color',palette(j,:),'MarkerSize',2.8);
                end
                xlabel(ax,'Paired platelet test'); ylabel(ax,'Normalized squared loss');
            case 8
                x=fromImage(fromImage.vector==vec,:);
                x.vs_G=nan(height(x),1); x.vs_R=x.vs_G;
                eligibleG=x.D_G>1e-12; eligibleR=x.D_R>1e-12;
                x.vs_G(eligibleG)=1-x.D_P(eligibleG)./x.D_G(eligibleG);
                x.vs_R(eligibleR)=1-x.D_P(eligibleR)./x.D_R(eligibleR);
                plot(ax,1:height(x),100*x.vs_G,'o','Color',palette(2,:),'MarkerSize',3);
                plot(ax,1:height(x),100*x.vs_R,'+','Color',palette(3,:),'MarkerSize',3);
                xlabel(ax,'Paired platelet test'); ylabel(ax,'Loss reduction (%)');
            case {9,10,11,13}
                keys=["certificate_gap","solve_seconds_per_block_batch_average","orthogonality","B_angle_target_error_success"];
                idx=find([9,10,11,13]==id);
                y=["Scaled trace gap","Time per block (s)","Orthogonality residual","Mean direction error (rad)"];
                x=tE(tE.metric==keys(idx),:);
                bars(ax,x,"mean",labels,palette,y(idx));
            case 12
                x=tA(tA.qf==0 & tA.comparison=="A_to_B" & startsWith(tA.metric,"delta_sigma_"),:);
                grouped(ax,x,labels,"metric","delta_sigma_"+(1:4),"Mean signed singular value change");
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
                    plot(ax,s.qf,s.ber_pct,'Color',[.68 .68 .68],'LineWidth',.55);
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
        case {6,12}
            details="Grouped bars show the metrics in each panel legend. S, G, R and P appear on the horizontal axes.";
        case 7
            details="S blue circle, G orange plus, R green triangle, P purple square. Infeasible G values are missing.";
        case 8
            details="Orange circles compare with G, green plus signs with R. Baseline loss must exceed 10^{-12}.";
        case 14
            details="Reselection and failure are measured over all 38,088 blocks for each G method.";
        case {15,22}
            details="S blue, G orange, R green, P purple. Point clouds include every calibration status.";
        case 24
            details="Grey curves are the eight Phase 02 methods. Coloured curves show Phase 03 groups meeting the PNG thresholds.";
    end
    xlabel(tl,details,'FontName','Times New Roman','FontSize',8);
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

function grouped(ax,x,labels,key,groups,label)
[found,idx]=ismember(x.(key),groups);
assert(all(found));
rows=methodIndex(x.method,labels);
v=nan(numel(labels),numel(groups));
v(sub2ind(size(v),rows,idx))=x.mean;
bar(ax,v);
xticks(ax,1:4); xticklabels(ax,extractBetween(labels,1,1)); ylabel(ax,label);
legend(ax,replace(groups,"_"," "),'Location','best','FontSize',6);
end
