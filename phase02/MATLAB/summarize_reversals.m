function summarize_reversals(runDir)
% 依原始嵌入成分的次序反轉旗標，比較後續 JPEG 的逐塊擾動。
arguments
    runDir (1,1) string
end
D=fullfile(runDir,'06_嵌入與擾動數據');
O=fullfile(runDir,'07_統計分析');
methods=["G0","S1","S2","S3","S4","S5","S6","S7"];
scopes=["Y","U","V","All"];
rows=cell(8*100*4*2,8);
rr=0;
for m=methods
    file=fullfile(D,m+"_embed.h5");
    flags=h5read(file,'/embedding/rank_reversal');
    if iscell(flags) || iscategorical(flags) || isstring(flags)
        flags=strcmpi(string(flags),'true');
    end
    flags=logical(flags(:));
    n=numel(flags)/3;
    math=fullfile(D,m+"_matlab.h5");
    for q=1:100
        path="/jpeg/"+compose("QF%03d",q)+"/B0_to_C";
        X=h5read(math,path,[1 1 1],[21 n 3]);
        X=reshape(X,21,[]).';
        for c=1:4
            ind=1:3*n;
            if c<4, ind=(c-1)*n+(1:n); end
            for status=[false,true]
                ix=ind(flags(ind)==status);
                rr=rr+1;
                rows(rr,:)={m,q,scopes(c),status,numel(ix),mean(X(ix,1)),mean(X(ix,5)), ...
                    mean(X(ix,[17 21]),'all')};
            end
        end
    end
    fprintf('%s reversal strata complete\n',m);
end
T=cell2table(rows,'VariableNames',{'method','qf','scope','rank_reversal','n','mean_D_F', ...
    'mean_nuclear_abs','mean_theta_4_sides'});
writetable(T,fullfile(O,'rank_reversal_strata.csv'));
end
