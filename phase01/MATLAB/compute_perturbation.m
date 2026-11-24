function [M,D,checks] = compute_perturbation(A,B,F,G,near)
% 同列索引進行角度及頻譜比較，保留所有原始配對。
E = B-A;
s = F.s;
sq = G.s;
inc = sq-s;
M.D_F = reshape(sum(E.^2,[1 2]),[],1);
M.D_sigma = sum(inc.^2,2);
M.R = M.D_F-M.D_sigma;
M.nuclear_original = sum(s,2);
M.nuclear_q = sum(sq,2);
M.nuclear_delta = M.nuclear_q-M.nuclear_original;
M.nuclear_abs = abs(M.nuclear_delta);
M.sigma = s;
M.sigma_q = sq;
M.delta_sigma = inc;
M.abs_delta_sigma = abs(inc);
M.theta_u = angles_svd(F.U,G.U);
M.theta_v = angles_svd(F.V,G.V);
[M.adjacent,M.gap,M.gap_norm] = spectral(s);
[M.adjacent_q,M.gap_q,M.gap_q_norm] = spectral(sq);
M.perturbation_over_gap = sqrt(M.D_F)./M.gap;
M.perturbation_over_gap(M.gap <= near*s(:,1)) = NaN;
a = flags(s,M.gap,near);
b = flags(sq,M.gap_q,near);
names = fieldnames(a);
D = struct();
for j = 1:numel(names)
    D.("original_"+names{j}) = a.(names{j});
    D.("jpeg_"+names{j}) = b.(names{j});
end
scale = max(1,reshape(sum(A.^2+B.^2,[1 2]),[],1));
df = zeros(size(M.D_F));
for j = 1:size(E,3)
    df(j) = norm(E(:,:,j),'fro')^2;
end
checks = [max(abs(M.D_F-df)./max(1,df)), max(max(0,-M.R)./scale)];
assert(checks(1) <= 32*eps && checks(2) <= 128*eps);
assert(all([M.theta_u M.theta_v] >= 0 & [M.theta_u M.theta_v] <= 90,'all'));
end

function [d,g,r] = spectral(s)
d = -diff(s,1,2);
g = [d(:,1) min(d(:,1:2),[],2) min(d(:,2:3),[],2) d(:,3)];
r = g./s(:,1);
end

function D = flags(s,g,near)
scale = s(:,1);
D.repeated = g <= 32*eps*scale;
D.near_repeated = g <= near*scale;
D.zero_singular = s <= 32*eps*scale;
D.near_rank_deficient = s(:,4) <= near*scale;
D.zero_matrix = scale == 0;
end
