function audit_figures(runDir)
% 實際重新開啟 FIG，確認是可取回座標軸和圖形物件的 MATLAB 圖檔。
arguments
    runDir (1,1) string
end
figureFolder = fullfile(runDir,'06_MATLAB圖表');
for id = 1:22
    filename = fullfile(figureFolder,sprintf('Fig%02d.fig',id));
    f = openfig(filename,'new','invisible');
    cleanup = onCleanup(@() close(f));
    axesList = findall(f,'Type','axes');
    assert(~isempty(axesList));
    assert(any(arrayfun(@(a) ~isempty(a.Children),axesList)));
    font = string({axesList.FontName});
    assert(all(contains(font,'Times New Roman','IgnoreCase',true)));
    fprintf('FIG%02d opened with %d axes using Times New Roman\n',id,numel(axesList));
    clear cleanup
end
end
