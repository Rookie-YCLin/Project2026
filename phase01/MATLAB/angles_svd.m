function theta = angles_svd(P,Q)
% 使用絕對內積，消除同一方向的正負號差異。
theta = reshape(acosd(min(1,max(0,abs(sum(P.*Q,1))))),4,[]).';
end
