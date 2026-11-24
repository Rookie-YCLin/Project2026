function F = factor(A)
% pagesvd 的 V 欄為右奇異向量，並檢查所有矩陣。
[U,S,V] = pagesvd(A);
s = reshape(S([1;6;11;16] + (0:size(A,3)-1)*16),4,[]).';
energy = @(X) reshape(sum(X.^2,[1 2]),[],1);
rec = pagemtimes(pagemtimes(U,S),'none',V,'transpose');
orth = @(X) max(sqrt(energy(pagemtimes(X,'transpose',X,'none')-eye(4))));
r = max(sqrt(energy(rec-A))./max(1,sqrt(energy(A))));
o = max(orth(U),orth(V));
assert(r <= 128*eps && o <= 128*eps);
assert(all(s>=0,'all') && all(diff(s,1,2)<=0,'all'));
F = struct('U',U,'s',s,'V',V,'reconstruction',r,'orthogonality',o);
end
