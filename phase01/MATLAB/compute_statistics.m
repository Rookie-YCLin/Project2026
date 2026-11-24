function [T,C,B,D,S] = compute_statistics(condition,M,names,X,flagNames,flags,n,near)
% 分群按原始區塊，All 直接使用三色通道合併後的觀測。
scopes = ["Y","U","V","All"];
features = ["gap","gap_norm","gap_q","gap_q_norm","perturbation_over_gap"];
edges = [0 1e-6 1e-4 1e-3 1e-2 1e-1 1 Inf];
stats = cell(4*numel(names),14);
corrRows = cell(4*4*5*2,8);
bins = cell(4*4*2*(numel(edges)-1),11);
diagRows = cell(4*numel(flagNames),5);
stableRows = cell(4*4*2,15);
eligible = M.gap > near*M.sigma(:,1) & M.gap_q > near*M.sigma_q(:,1) & ...
    M.sigma > near*M.sigma(:,1) & M.sigma_q > near*M.sigma_q(:,1);
iT = 0; iC = 0; iB = 0; iD = 0; iS = 0;
for c = 1:4
    ix = 1:3*n;
    if c < 4
        ix = (c-1)*n+(1:n);
    end
    for j = 1:numel(names)
        iT = iT+1;
        stats(iT,:) = [{condition,scopes(c),names(j)},num2cell(distribution(X(ix,j)))];
    end
    for j = 1:numel(flagNames)
        iD = iD+1;
        diagRows(iD,:) = {condition,scopes(c),flagNames(j),numel(ix),sum(flags(ix,j))};
    end
    for j = 1:4
        g = M.gap_norm(ix,j);
        for side = ["u","v"]
            a = M.("theta_"+side);
            y = a(ix,j);
            iS = iS+1;
            stableRows(iS,:) = [{condition,scopes(c),j,side},num2cell(distribution(y(eligible(ix,j))))];
            for f = features
                x = M.(f);
                x = x(ix,j);
                mask = isfinite(x)&isfinite(y);
                xx = x(mask);
                yy = y(mask);
                p = NaN; s = NaN;
                if numel(xx)>1 && max(xx)>min(xx) && max(yy)>min(yy)
                    r = corrcoef(xx,yy); p = r(1,2);
                    r = corrcoef(midrank(xx),midrank(yy)); s = r(1,2);
                end
                iC = iC+1;
                corrRows(iC,:) = {condition,scopes(c),j,side,f,numel(xx),p,s};
            end
            for b = 1:numel(edges)-1
                v = distribution(y(g>=edges(b) & g<edges(b+1)));
                iB = iB+1;
                bins(iB,:) = {condition,scopes(c),j,side,b,edges(b),edges(b+1),v(1),v(3),v(7),v(10)};
            end
        end
    end
end
T = cell2table(stats,'VariableNames',{'condition','scope','metric','count','mean','median','std','min','max','p10','p25','p75','p90','p95'});
C = cell2table(corrRows,'VariableNames',{'condition','scope','j','side','feature','count','pearson','spearman'});
B = cell2table(bins,'VariableNames',{'condition','scope','j','side','bin','lower','upper','count','median','p10','p90'});
D = cell2table(diagRows,'VariableNames',{'condition','scope','flag','count','true_count'});
S = cell2table(stableRows,'VariableNames',{'condition','scope','j','side','count','mean','median','std','min','max','p10','p25','p75','p90','p95'});
end

function T = distribution(x)
y = sort(x(isfinite(x)));
n = numel(y);
if n == 0
    T = [0 nan(1,10)];
    return
end
h = 1 + (n-1)*[0.50 0.10 0.25 0.75 0.90 0.95];
a = floor(h);
b = min(a+1,n);
q = reshape(y(a),1,[]) + (h-a).*(reshape(y(b),1,[])-reshape(y(a),1,[]));
T = [n mean(y) q(1) std(y,1) y(1) y(end) q(2:end)];
end

function ranks = midrank(x)
[v,ix] = sort(x(:));
start = [1; find(diff(v)~=0)+1];
stop = [start(2:end)-1; numel(v)];
ranks = zeros(size(v));
ranks(ix) = repelem((start+stop)/2,stop-start+1);
end
