function [S,G,R,planes,jG,thetaG,thetaR] = phase03_simple_candidates(w,c,j,phi)
% S 與 R 的最近目標方向以及在相同角度的 G 平面枚舉。
I=eye(4);
e=I(:,j);
t=cos(phi);
if phi==pi/2, t=0; end
if c(j)<0, t=-t; end
z=e-c(j)*c;
if norm(z)<=16*eps
    [~,k]=min(abs(c));
    z=I(:,k)-c(k)*c;
end
x=t*c+sin(phi)*z/norm(z);
S=I;
S(:,j)=x;
u=x-e*x(j);
ss=norm(u);
thetaR=atan2(ss,x(j));
R=I;
if ss>8*eps
    u=u/ss;
    R=I+(x(j)-1)*(e*e.'+u*u.')+ss*(u*e.'-e*u.');
end
G=[];
planes=0;
best=Inf;
jG=NaN;
thetaG=NaN;
for k=setdiff(1:4,j)
    norm2=hypot(c(j),c(k));
    residual=norm(c(setdiff(1:4,[j,k])));
    if phi+16*eps<atan2(residual,norm2), continue; end
    planes=planes+1;
    if norm2==0
        angles=0;
    else
        p=atan2(c(k),c(j));
        t=abs(t);
        beta=atan2(sqrt(max(0,(sin(phi)-residual)*(sin(phi)+residual))),t);
        angles=[p+beta,p-beta,p+beta+pi,p-beta+pi];
    end
    for theta=angles
        x=I;
        p=cos(theta); z=sin(theta);
        x([j k],[j k])=[p -z; z p];
        F=sum((x-I).^2.*w.','all');
        if F<best
            best=F; G=x; jG=k; thetaG=abs(atan2(z,p));
        end
    end
end
end
