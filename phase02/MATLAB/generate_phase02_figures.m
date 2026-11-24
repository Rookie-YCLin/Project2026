function generate_phase02_figures(runDir, ids, figDir)
% 讀取 MATLAB 主統計與盲式 BER，重建任意圖號。
arguments
    runDir (1,1) string
    ids (1,:) double = 1:18
    figDir (1,1) string = fullfile(runDir,'08_MATLAB圖表')
end
D=fullfile(runDir,'07_統計分析');
models=["G0","S1","S2","S3","S4","S5","S6","S7"];
color=[0.12 0.12 0.12;0.00 0.34 0.65;0.83 0.32 0.05;0.00 0.46 0.32;0.60 0.30 0.70; ...
       0.10 0.55 0.68;0.66 0.47 0.06;0.58 0.19 0.28];
style=["-","--","-.",":","-","--","-.",":"];
K=cell(8,5);
for k=1:8
    m=models(k);
    endings=["ber_psnr","matrix_stats","allocation_stats","decision_stats","spectral_stats"];
    for i=1:numel(endings)
        K{k,i}=readtable(fullfile(D,m+"_"+endings(i)+".csv"),'TextType','string');
        K{k,i}.method=repmat(m,height(K{k,i}),1);
    end
end
cal=readtable(fullfile(runDir,'02_步長校準',"selected.csv"),'TextType','string');
C=vertcat(K{:,1});
C=C(C.condition~="unattacked",:);
A=vertcat(K{:,2}); Alloc=vertcat(K{:,3}); Q=vertcat(K{:,4}); Sp=vertcat(K{:,5});
for i=1:numel(ids)
    id=ids(i);
    filename=fullfile(figDir,compose('Fig%02d',id));
    assert(~isfile(filename+".fig"));
    f=figure('Visible','off','Color','white','Units','inches','Position',[1 1 6.5 3.5+2.4*ismember(id,[3 9 13 15])], ...
        'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman','DefaultAxesFontSize',10);
    cleanup=onCleanup(@() close(f));
    switch id
        case 1
            T=C; ax=axes(f); lines(ax,T,"ber_pct",models,color,style,"Final BER (%)","Final watermark BER",100);
        case 2
            T=C; ax=axes(f); lines(ax,T,"psnr_original_db",models,color,style,"PSNR (dB)","PSNR relative to the original image",100);
        case 3
            T=cal;
            tl=tiledlayout(f,2,1,'TileSpacing','compact');
            a=nexttile(tl);
            groups=cal.method~="G0";
            bar(a,categorical(cal.method(groups)),cal.parameter(groups),'FaceColor',color(2,:));
            ylabel(a,"Global sum-QIM step q");
            g=cal.parameter(cal.method=="G0");
            title(a,compose("G0 parameters: alpha = %.7g, d1 = %.7g, d2 = %.7g",g,36*g,20*g));
            a=nexttile(tl);
            bar(a,categorical(cal.method),cal.psnr_db-40,'FaceColor',color(4,:));
            ylim(a,[-.105 .105]); yline(a,.1,':'); yline(a,-.1,':');
            ylabel(a,"PNG PSNR minus 40 dB");
        case 4
            T=A(A.scope=="All"&A.metric=="D_F"&A.comparison=="A_to_B"&A.method~="G0",:);
            bars(axes(f),T,models(2:end),"comparison","mean","Mean squared Frobenius norm","Theoretical embedding perturbation");
        case 5
            T=A(A.scope=="All"&A.metric=="D_F"&ismember(A.comparison,["A_to_B","A_to_B0","B_to_B0"]),:);
            bars(axes(f),T,models,"comparison","mean","Mean squared Frobenius norm","Theoretical and realized perturbations");
        case 6
            T=Alloc(Alloc.scope=="All"&ismember(Alloc.metric,["nonnegative_trigger","projected","reselected"]),:);
            T.percent=100*T.mean;
            bars(axes(f),T,models(2:end),"metric","percent","Blocks (%)","Nonnegativity and target reselection");
        case 7
            T=Alloc(Alloc.scope=="All"&ismember(Alloc.metric,["rank_reversal","adjacent_tie"]),:);
            T.percent=100*T.mean;
            bars(axes(f),T,models,"metric","percent","Blocks (%)","Reversals and numerical ties");
        case 8
            T=series(A,"D_F");
            lines(axes(f),T,"mean",models,color,style,"Mean squared Frobenius norm","JPEG-only matrix perturbation",100);
        case 9
            T=vertcat(series(A,"abs_delta_1"),series(A,"abs_delta_2"),series(A,"abs_delta_3"),series(A,"abs_delta_4"));
            tl=tiledlayout(f,2,2,'TileSpacing','compact');
            for j=1:4
                ax=nexttile(tl);
                lines(ax,T(T.metric=="abs_delta_"+j,:),"mean",models,color,style,"Mean absolute change","j = "+j,100);
            end
        case 10
            T=series(A,"nuclear_delta");
            ax=axes(f); lines(ax,T,"mean",models,color,style,"Mean signed change","JPEG-only nuclear norm change",100);
            yline(ax,0,':','HandleVisibility','off');
        case 11
            T=series(A,"nuclear_abs");
            lines(axes(f),T,"mean",models,color,style,"Mean absolute change","Absolute JPEG-only nuclear norm change",100);
        case 12
            T=Q(Q.scope=="All"&Q.metric=="z"&Q.condition~="unattacked",:);
            T.qf=str2double(extractAfter(T.condition,2));
            ax=axes(f); lines(ax,T,"mean",models(2:end),color(2:end,:),style(2:end),"Mean normalized offset e / q","Sum-QIM center offset",100);
            yline(ax,0,':','HandleVisibility','off');
        case 13
            T=Q(Q.scope=="All"&ismember(Q.metric,["target_margin","block_error"])&Q.condition~="unattacked",:);
            T.qf=str2double(extractAfter(T.condition,2));
            T.block_correct_pct=100*(1-T.mean);
            tl=tiledlayout(f,2,1,'TileSpacing','compact');
            lines(nexttile(tl),T(T.metric=="target_margin",:),"mean",models(2:end),color(2:end,:),style(2:end), ...
                "Mean margin","Target margin",100);
            lines(nexttile(tl),T(T.metric=="block_error",:),"block_correct_pct",models(2:end),color(2:end,:),style(2:end), ...
                "Correct blocks (%)","Actual parity decision",100);
        case 14
            T=C;
            T.block_ber_pct=100*T.block_ber;
            ax=axes(f);
            scatter(ax,T.block_ber_pct,T.ber_pct,11,T.qf,"filled");
            xlabel(ax,"Block BER (%)"); ylabel(ax,"Final BER (%)"); title(ax,"Block decision and final payload errors");
            b=colorbar(ax); b.Label.String="JPEG QF"; grid(ax,'on');
        case 15
            T=Sp(Sp.scope=="All"&ismember(Sp.condition,["A","B0"])&ismember(Sp.metric,["gap12","gap23","gap34"]),:);
            T.key=T.condition+" "+T.metric;
            tl=tiledlayout(f,3,1,'TileSpacing','compact');
            metrics=["gap12","gap23","gap34"];
            subtitles=["Gap between 1 and 2","Gap between 2 and 3","Gap between 3 and 4"];
            for j=1:3
                bars(nexttile(tl),T(T.metric==metrics(j),:),models,"condition","median","Median gap",subtitles(j));
            end
            title(tl,"Adjacent gaps before and after embedding",'FontName','Times New Roman');
        case 16
            T=representative(runDir);
            ax=axes(f);
            fn=["A","B","B0","QF080"];
            hold(ax,'on');
            for j=1:4
                plot(ax,1:4,T.(fn(j)),'-o','Color',color(j,:),'DisplayName',fn(j),'LineWidth',1.4);
            end
            set(ax,'YScale','log','XTick',1:4);
            xlabel(ax,"Singular index"); ylabel(ax,"Singular value");
            title(ax,"S5: central Y block, spectra through the four stages");
            legend(ax,'Location','northoutside','Orientation','horizontal'); grid(ax,'on');
        case 17
            T=C(C.qf<=30,:);
            lines(axes(f),T,"ber_pct",models,color,style,"Final BER (%)","Low-QF region",30);
        case 18
            baseline=sortrows(C(C.method=="G0",:),"qf");
            T=sortrows(C(C.method~="G0",:),["method","qf"]);
            T.error_count_minus_G0=T.bit_errors-repmat(baseline.bit_errors,7,1);
            ax=axes(f);
            lines(ax,T,"error_count_minus_G0",models(2:end),color(2:end,:),style(2:end),"Difference in error count","Errors relative to G0",100);
            yline(ax,0,':','HandleVisibility','off');
        otherwise
            error('Unknown figure ID')
    end
    writetable(T,filename+".csv");
    savefig(f,filename+".fig");
    exportgraphics(f,filename+".svg",'BackgroundColor','white');
    exportgraphics(f,filename+".pdf",'ContentType','vector','BackgroundColor','white');
    exportgraphics(f,filename+".png",'Resolution',350,'BackgroundColor','white');
    fprintf('Phase02 Fig%02d exported\n',id);
    clear cleanup
end
end

function T=series(A,name)
T=A(A.scope=="All"&A.metric==name&A.comparison=="B0_to_C",:);
T.qf=str2double(extractAfter(T.condition,2));
end

function lines(ax,T,field,models,colors,styles,yl,heading,last)
hold(ax,'on');
for j=1:numel(models)
    X=sortrows(T(T.method==models(j),:),"qf");
    plot(ax,X.qf,X.(field),'Color',colors(j,:),'LineStyle',styles(j),'LineWidth',1.35,'DisplayName',models(j));
end
title(ax,heading); ylabel(ax,yl); xlabel(ax,"JPEG quality factor (QF)");
xlim(ax,[1 last]);
legend(ax,'Location','northoutside','Orientation','horizontal','NumColumns',4);
grid(ax,'on'); box(ax,'on');
end

function bars(ax,T,models,group,field,yl,heading)
groups=unique(T.(group),'stable');
D=nan(numel(models),numel(groups));
for j=1:numel(models)
    for k=1:numel(groups)
        hit=T.method==models(j)&T.(group)==groups(k);
        if any(hit), D(j,k)=T.(field)(hit); end
    end
end
bar(ax,categorical(models,models),D,'grouped');
ylabel(ax,yl); title(ax,heading);
legend(ax,replace(groups,"_"," "),'Location','northoutside','Orientation','horizontal','NumColumns',3);
grid(ax,'on'); box(ax,'on');
end

function T=representative(run)
input=fullfile(run,'05_原始矩陣','common.h5');
d=double(h5read(input,'/index/grid'));
i=floor(d(1)/2)*d(2)+floor(d(2)/2)+1;
src=fullfile(run,'06_嵌入與擾動數據');
A=h5read(fullfile(src,'G0_matlab.h5'),'/original/sigma',[1 i 1],[4 1 1]);
names=["B","B0","QF080"];
v=zeros(4,3);
for j=1:3
    key="/"+names(j)+"/sigma";
    if j==3, key="/jpeg"+key; end
    v(:,j)=h5read(fullfile(src,'S5_matlab.h5'),key,[1 i 1],[4 1 1]);
end
T=array2table([(1:4).',repmat(i-1,4,1),A,v], ...
    'VariableNames',{'j','block_index0','A','B','B0','QF080'});
end
