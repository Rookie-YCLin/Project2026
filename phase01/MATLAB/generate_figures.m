function generate_figures(runDir, ids, figDir)
% 依圖號重建任意正式圖，資料來源為 MATLAB 的逐塊或統計輸出。
arguments
    runDir (1,1) string
    ids (1,:) double = 1:22
    figDir (1,1) string = fullfile(runDir,'06_MATLAB圖表')
end
assert(all(ismember(ids,1:22)) && numel(unique(ids))==numel(ids));
if ~isfolder(figDir), mkdir(figDir); end
statsDir = fullfile(runDir,'05_統計分析');
S = readtable(fullfile(statsDir,'statistics_matlab.csv'),'TextType','string');
S = S(S.condition~="identity",:);
S.QF = str2double(extractAfter(S.condition,2));
scopes = ["Y","U","V","All"];
components = ["j = 1","j = 2","j = 3","j = 4"];
titles = ["Mean squared Frobenius perturbation", ...
    "Absolute singular-value change", "Left singular-vector angle", "Right singular-vector angle"];
prefixes = ["D_F", "abs_delta_sigma", "theta_u", "theta_v"];
yLabels = ["Mean squared Frobenius norm", "Mean absolute change", "Mean angle (degree)", "Mean angle (degree)"];
metadata = ["Squared Frobenius perturbation", ...
    "Absolute singular-value perturbation — Y", "Absolute singular-value perturbation — U", ...
    "Absolute singular-value perturbation — V", "Absolute singular-value perturbation — All", ...
    "Left angle — Y", "Left angle — U", "Left angle — V", "Left angle — All", ...
    "Right angle — Y", "Right angle — U", "Right angle — V", "Right angle — All", ...
    "Signed singular-value perturbation", "Nuclear norm perturbation", ...
    "Squared Frobenius and singular-value difference", "Normalized gap distribution", ...
    "Normalized gap and left angle", "Normalized gap and right angle", ...
    "Spatial distribution", "Low-QF detail", "Perturbation percentile bands"];
colors = [0.05 0.32 0.62; 0.73 0.29 0.08; 0.03 0.46 0.35; 0.48 0.24 0.54];
styles = ["-","--","-.",":"];
cfg = jsondecode(fileread(fullfile(fileparts(fileparts(mfilename('fullpath'))),'Configs','phase01.json')));
for id = ids
    label = sprintf('Fig%02d',id);
    out = fullfile(figDir,label);
    assert(~isfile(out+".fig"),'圖檔已存在，請指定新的圖表資料夾。');
    tall = ismember(id,[14 17 18 19 20]);
    f = figure('Visible','off','Color','white','Units','inches', ...
        'Position',[1 1 6.4 3.45+3.65*tall],'InvertHardcopy','off', ...
        'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman', ...
        'DefaultAxesFontSize',10);
    cleanup = onCleanup(@() close(f));
    if id == 1
        ax = axes(f);
        T = S(S.metric=="D_F",:);
        linesFromStats(ax,T,"scope",scopes,scopes,"mean",colors,styles);
        decorate(ax,titles(1),yLabels(1),100);
    elseif id <= 13
        ax = axes(f);
        family = floor((id-2)/4)+2;
        c = mod(id-2,4)+1;
        fields = prefixes(family)+"_"+(1:4);
        T = S(ismember(S.metric,fields)&S.scope==scopes(c),:);
        linesFromStats(ax,T,"metric",fields,components,"mean",colors,styles);
        decorate(ax,titles(family)+" ("+scopes(c)+")",yLabels(family),100);
    elseif id == 14
        tl = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
        fields = "delta_sigma_"+(1:4);
        T = S(ismember(S.metric,fields),:);
        for c = 1:4
            ax = nexttile(tl);
            linesFromStats(ax,T(T.scope==scopes(c),:),"metric",fields,components,"mean",colors,styles);
            yline(ax,0,'Color',[.56 .56 .56],'HandleVisibility','off');
            decorate(ax,scopes(c),"Mean signed change",100);
        end
        title(tl,"Signed singular-value perturbation",'FontName','Times New Roman');
    elseif id == 15
        ax = axes(f);
        T = S(S.scope=="All"&ismember(S.metric,["nuclear_delta","nuclear_abs"]),:);
        linesFromStats(ax,T,"metric",["nuclear_delta","nuclear_abs"],["Signed","Absolute"],"mean",colors,styles);
        decorate(ax,"Nuclear norm perturbation (All)","Mean change",100);
    elseif id == 16
        ax = axes(f);
        fields = ["D_F","D_sigma","R"];
        T = S(S.scope=="All"&ismember(S.metric,fields),:);
        linesFromStats(ax,T,"metric",fields,["D_F","D_{sigma}","R"],"mean",colors,styles);
        decorate(ax,"Comparison of squared perturbations (All)","Mean squared difference",100);
    elseif id == 17
        tl = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
        fields = "gap_q_norm_"+(1:4);
        T = S(ismember(S.metric,fields),:);
        for c = 1:4
            ax = nexttile(tl);
            hold(ax,'on');
            floorGap = 1e-8;
            for j = 1:4
                C = sortrows(T(T.scope==scopes(c)&T.metric==fields(j),:),'QF');
                x = C.QF(:);
                band = [max(floorGap,C.p10(:)); flipud(max(floorGap,C.p90(:)))];
                fill(ax,[x;flipud(x)],band,colors(j,:),'FaceAlpha',.13, ...
                    'EdgeColor','none','HandleVisibility','off');
            end
            for j = 1:4
                C = sortrows(T(T.scope==scopes(c)&T.metric==fields(j),:),'QF');
                plot(ax,C.QF,max(floorGap,C.median),'Color',colors(j,:),'LineStyle',styles(j), ...
                    'LineWidth',1.25,'DisplayName',components(j));
            end
            legend(ax,'Location','northoutside','Orientation','horizontal','NumColumns',4);
            set(ax,'YScale','log');
            ylim(ax,[5e-9 1.4]);
            yticks(ax,[1e-8 1e-6 1e-4 1e-2 1]);
            decorate(ax,scopes(c),"Normalized nearest gap",100);
        end
        title(tl,["Normalized nearest gaps after JPEG (median and P10–P90)", ...
            "Values below 10^{-8} are shown at 10^{-8}"],'FontName','Times New Roman');
    elseif ismember(id,[18 19])
        side = "u";
        if id == 19, side = "v"; end
        B = readtable(fullfile(statsDir,'gap_bins_matlab.csv'),'TextType','string');
        T = B(B.side==side & B.scope=="All" & ismember(B.condition,compose("QF%03d",cfg.figure_qf)),:);
        T.QF = str2double(extractAfter(T.condition,2));
        tl = tiledlayout(f,numel(cfg.figure_qf),1,'TileSpacing','compact','Padding','compact');
        for q = cfg.figure_qf(:).'
            ax = nexttile(tl);
            hold(ax,'on');
            for j = 1:4
                C = sortrows(T(T.QF==q&T.j==j,:),'bin');
                x = C.bin + (j-2.5)*.09;
                errorbar(ax,x,C.median,C.median-C.p10,C.p90-C.median,'o', ...
                    'Color',colors(j,:),'LineWidth',1,'MarkerSize',4,'DisplayName',components(j));
            end
            ylim(ax,[0 90]); xlim(ax,[.55 7.45]);
            xticks(ax,1:7);
            xticklabels(ax,["[0, 10^{-6})","[10^{-6}, 10^{-4})","[10^{-4}, 10^{-3})", ...
                "[10^{-3}, 10^{-2})","[10^{-2}, 0.1)","[0.1, 1)","[1, inf)"]);
            xlabel(ax,"Original normalized nearest gap");
            ylabel(ax,"Angle (degree)");
            title(ax,"QF "+q);
            legend(ax,'Location','northoutside','Orientation','horizontal');
            grid(ax,'on');
        end
        title(tl,metadata(id)+" (All, median and P10–P90)",'FontName','Times New Roman');
    elseif id == 20
        [T,tiles,gridSize] = heatmaps(runDir,cfg.figure_qf);
        tl = tiledlayout(f,3,3,'TileSpacing','compact','Padding','compact');
        top = max(log10(1+T.D_F));
        for k = 1:9
            ax = nexttile(tl);
            imagesc(ax,tiles{k},[0 top]);
            axis(ax,'image');
            colormap(ax,parula(256));
            xlabel(ax,"Block column"); ylabel(ax,"Block row");
            title(ax,scopes(mod(k-1,3)+1)+", QF "+cfg.figure_qf(ceil(k/3)));
            xticks(ax,[1 gridSize(2)]); yticks(ax,[1 gridSize(1)]);
        end
        bar = colorbar(ax);
        bar.Layout.Tile = 'east';
        bar.Label.String = 'log_{10}(1 + D_F)';
        title(tl,"Spatial distribution of squared perturbation",'FontName','Times New Roman');
    elseif id == 21
        ax = axes(f);
        T = S(S.metric=="D_F"&S.QF<=20,:);
        linesFromStats(ax,T,"scope",scopes,scopes,"mean",colors,styles);
        decorate(ax,"Low-QF detail","Mean squared Frobenius norm",20);
    else
        ax = axes(f);
        T = sortrows(S(S.metric=="D_F"&S.scope=="All",:),'QF');
        x = T.QF;
        hold(ax,'on');
        fill(ax,[x;flipud(x)],[T.p10;flipud(T.p90)],[.86 .91 .96], ...
            'EdgeColor','none','DisplayName','P10–P90');
        fill(ax,[x;flipud(x)],[T.p25;flipud(T.p75)],[.66 .80 .90], ...
            'EdgeColor','none','DisplayName','P25–P75');
        plot(ax,x,T.median,'Color',colors(1,:),'LineWidth',1.5,'DisplayName','Median');
        legend(ax,'Location','northoutside','Orientation','horizontal');
        decorate(ax,"Distribution of squared perturbation (All)","Squared Frobenius norm",100);
    end
    writetable(T,out+".csv");
    savefig(f,out+".fig");
    exportgraphics(f,out+".svg",'BackgroundColor','white');
    exportgraphics(f,out+".pdf",'ContentType','vector','BackgroundColor','white');
    exportgraphics(f,out+".png",'Resolution',350,'BackgroundColor','white');
    fprintf('Exported %s: %s\n',label,metadata(id));
    clear cleanup
end
end

function linesFromStats(ax,T,key,values,labels,stat,colors,styles)
hold(ax,'on');
for i = 1:numel(values)
    X = sortrows(T(T.(key)==values(i),:),'QF');
    plot(ax,X.QF,X.(stat),'LineWidth',1.45,'Color',colors(i,:),'LineStyle',styles(i), ...
        'DisplayName',labels(i));
end
legend(ax,'Location','northoutside','Orientation','horizontal','NumColumns',min(numel(values),4));
end

function decorate(ax,t,y,last)
xlabel(ax,"JPEG quality factor (QF)");
ylabel(ax,y);
title(ax,t);
xlim(ax,[1 last]);
if last == 100, xticks(ax,[1 20 40 60 80 100]); else, xticks(ax,[1 5 10 15 20]); end
grid(ax,'on');
box(ax,'on');
end

function [T,tiles,grid] = heatmaps(runDir,quality)
A = fullfile(runDir,'02_原始矩陣','block_matrices.h5');
M = fullfile(runDir,'04_逐區塊擾動數據','perturbation_matlab.h5');
grid = double(h5read(A,'/index/grid'));
n = prod(grid);
data = cell(numel(quality)*3,1);
tiles = data;
[r,c] = ndgrid(1:grid(2),1:grid(1));
scopes = ["Y","U","V"];
for i = 1:numel(quality)
    path = "/jpeg/"+compose("QF%03d",quality(i))+"/metrics";
    names = split(string(h5readatt(M,path,'fields')),",");
    x = reshape(h5read(M,path,[find(names=="D_F") 1 1],[1 n 3]),n,3);
    for j = 1:3
        k = (i-1)*3+j;
        tiles{k} = reshape(log10(1+x(:,j)),grid(2),grid(1)).';
        data{k} = table(repmat(quality(i),n,1),repmat(scopes(j),n,1),c(:),r(:),x(:,j), ...
            'VariableNames',{'QF','scope','block_row','block_column','D_F'});
    end
end
T = vertcat(data{:});
end
