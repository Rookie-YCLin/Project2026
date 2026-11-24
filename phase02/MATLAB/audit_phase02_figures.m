function audit_phase02_figures(runDir,ids)
% 重新開啟每個 FIG 以確認原始座標軸可編輯。
if nargin<2, ids=1:18; end
for id=ids
    path=fullfile(runDir,'08_MATLAB圖表',compose('Fig%02d.fig',id));
    f=openfig(char(path),'new','invisible');
    cleanup=onCleanup(@() close(f));
    a=findall(f,'Type','axes');
    assert(~isempty(a));
    assert(all(contains(string({a.FontName}),'Times New Roman','IgnoreCase',true)));
    fprintf('Fig%02d verified with %d axes\n',id,numel(a));
    clear cleanup
end
end
