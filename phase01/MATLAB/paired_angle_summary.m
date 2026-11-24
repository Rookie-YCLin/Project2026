function paired_angle_summary(runDir)
% 從 MATLAB 已保存的逐區塊角度計算配對統計與有號差方向比例。
arguments
    runDir (1,1) string
end
file = fullfile(runDir,'04_逐區塊擾動數據','perturbation_matlab.h5');
out = fullfile(runDir,'05_統計分析','paired_angles_and_signs_matlab.csv');
corrOut = fullfile(runDir,'05_統計分析','identifiable_gap_correlations_matlab.csv');
assert(~isfile(out) && ~isfile(corrOut));
cfg = jsondecode(fileread(fullfile(fileparts(fileparts(mfilename('fullpath'))),'Configs','phase01.json')));
scopes = ["Y","U","V","All"];
info = h5info(file,'/jpeg');
rows = cell(16*numel(info.Groups),12);
r = 0;
cs = cell(32*numel(info.Groups),7);
rc = 0;
for k = 1:numel(info.Groups)
    g = string(info.Groups(k).Name);
    label = extractAfter(g,"/jpeg/");
    path = g+"/metrics";
    fields = split(string(h5readatt(file,path,'fields')),",");
    data = h5read(file,path);
    n = size(data,2);
    for j = 1:4
        keys = ["theta_u_"+j,"theta_v_"+j,"delta_sigma_"+j,"sigma_1","sigma_q_1", ...
            "gap_"+j,"gap_q_"+j,"sigma_"+j,"sigma_q_"+j,"gap_norm_"+j,"perturbation_over_gap_"+j];
        xi = zeros(size(keys));
        for i = 1:numel(keys)
            xi(i) = find(fields==keys(i),1);
        end
        X = reshape(data(xi,:,:),numel(keys),[]);
        tol = 32*eps*max(X(4,:),X(5,:));
        leftScale = cfg.relative_near_threshold*X(4,:);
        rightScale = cfg.relative_near_threshold*X(5,:);
        stable = X(6,:)>leftScale & X(7,:)>rightScale & X(8,:)>leftScale & X(9,:)>rightScale;
        for c = 1:4
            idx = 1:3*n;
            if c<4, idx = (c-1)*n+(1:n); end
            u = X(1,idx); v = X(2,idx); inc = X(3,idx); z = tol(idx);
            p = NaN;
            if max(u)>min(u) && max(v)>min(v)
                C = corrcoef(u,v);
                p = C(1,2);
            end
            r = r+1;
            rows(r,:) = {label,scopes(c),j,numel(idx),mean(u),mean(v),mean(abs(u-v)), ...
                p,sum(inc>z),sum(inc< -z),sum(abs(inc)<=z),median(inc)};
            keep = idx(stable(idx));
            sides = ["u","v"];
            for side = 1:2
                rc = rc+1;
                cs(rc,:) = {label,scopes(c),j,sides(side),numel(keep), ...
                    correlation(X(10,keep),X(side,keep)),correlation(X(11,keep),X(side,keep))};
            end
        end
    end
end
T = cell2table(rows,'VariableNames',{'condition','scope','j','count','left_mean','right_mean', ...
    'mean_absolute_paired_angle_difference','pearson_left_right','positive','negative','near_zero','median_signed'});
writetable(T,out);
C = cell2table(cs,'VariableNames',{'condition','scope','j','side','count', ...
    'pearson_gap_norm','pearson_perturbation_over_gap'});
writetable(C,corrOut);
end

function r = correlation(x,y)
r = NaN;
if numel(x)>1 && max(x)>min(x) && max(y)>min(y)
    C = corrcoef(x,y); r = C(1,2);
end
end
