function checks = validate_numerical_results()
% 使用對角、非對稱、零、秩一、重根與近重根矩陣做正反例。
A = cat(3,diag([4 3 2 1]),[5 1 -2 3;2 4 0 1;1 2 7 4;0 0 1 3], ...
    zeros(4),[1;2;3;4]*[4 3 2 1],diag([5 3 3 1]),diag([5 3+1e-8 3 1]));
F = factor(A);
t = angles_svd(F.U,-F.U);
assert(max(t,[],'all') <= 4e-6);
assert(isequal(angles_svd(F.U,F.U),t));
assert(max(abs(F.s(1,:)-[4 3 2 1]))<=eps);
[M,D] = compute_perturbation(A,A,F,F,1e-6);
assert(all(M.D_F==0 & M.D_sigma==0) && all(M.nuclear_delta==0));
assert(D.original_zero_matrix(3) && all(D.original_zero_singular(4,2:4)));
assert(D.original_repeated(5,2) && ~D.original_repeated(6,2) && D.original_near_repeated(6,2));
assert(D.original_near_rank_deficient(4) && isnan(M.gap_norm(3,1)));
[Y,~] = compute_perturbation(A(:,:,1),diag([5 3 2 1]),factor(A(:,:,1)),factor(diag([5 3 2 1])),1e-6);
assert(Y.D_F==1 && Y.D_sigma==1 && Y.nuclear_delta==1);
[Z,~] = compute_perturbation(A(:,:,1),A(:,:,1)*circshift(eye(4),[0 1]), ...
    factor(A(:,:,1)),factor(A(:,:,1)*circshift(eye(4),[0 1])),1e-6);
assert(Z.D_sigma==0 && Z.D_F>0 && all(Z.theta_v==90));
checks = struct('passed',true,'matrix_count',6,'max_reconstruction',F.reconstruction, ...
    'max_orthogonality',F.orthogonality,'sign_check_max_degree',max(t,[],'all'));
end
