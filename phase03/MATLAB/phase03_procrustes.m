function branches = phase03_procrustes(w,c,j,phi)
% 以 4 個線性等式與行列式分支求解，逐分支保存可行值與對偶上界。
branches=cell(4,1);
k=0;
for tau=[1 -1]
    for s=[1 -1]
        k=k+1;
        branches{k}=solve_branch(w,c,j,phi,tau,s);
    end
end
branches=vertcat(branches{:});
end

function b=solve_branch(w,c,j,phi,tau,s)
t=tau*cos(phi);
if phi==pi/2, t=0; end
ss=sin(phi);
dim=numel(c);
I=eye(dim);
energy=sum(w);
if phi==0 || energy==0
    x=direction(c,I(:,j),t,ss);
    R=completion(w,x,j,s);
    f=w.'*diag(R);
    name="pole";
    if energy==0, name="zero_objective"; end
    b=make_result(R,f,f,0,s,t,name,0);
    return
end
history=cell(140,1);
n=0;
low=-1; high=1;
dv0=record(low);
dv1=record(high);
while dv0>0 || dv1<0
    if n>110, error("Could not bracket an interior dual minimum."); end
    if dv0>0
        low=2*low; dv0=record(low);
    end
    if dv1<0
        high=2*high; dv1=record(high);
    end
end
for k=1:95
    center=low+(high-low)/2;
    der=record(center);
    if abs(der)<=2*eps || high-low<=4*eps*max(realmin,abs(center)), break; end
    if der<0, low=center; else, high=center; end
end
points=vertcat(history{1:n});
[upper,index]=sort([points.dual]);
[~,stationary]=min(abs([points.slope]));
% At a kink, rounded dual values may tie along an almost flat segment.
% Always inspect the final spectral states at the narrowed bracket.
selected=unique([max(1,n-4):n, index(1:min(4,n)), stationary],'stable');
best=-Inf;
for idx=selected
    z=points(idx);
    for m=[16,64,256,1024,1e5,1e7]
        [col,kind]=support_column(z,c,j,t,m);
        R=completion(w,direction(c,col,t,ss),j,s);
        value=w.'*diag(R);
        if value>best
            P=R; best=value; multiplier=z.lambda; face=kind;
        end
        if upper(1)-best<=5e-12*max(1,energy), break; end
    end
    if upper(1)-best<=5e-12*max(1,energy), break; end
end
b=make_result(P,best,upper(1),multiplier,s,t,face,n);
    function der=record(lam)
        M=diag(w);
        M(:,j)=M(:,j)+lam*c;
        [L,sv,H]=svd(M,"vector");
        correction=s*sign(det(L*H.'));
        J=L;
        J(:,end)=correction*J(:,end);
        Q=J*H.';
        der=c.'*Q(:,j)-t;
        dual=sum(sv(1:end-1))+correction*sv(end)-lam*t;
        dual=dual+32*eps*(1+sum(w)+abs(lam));
        n=n+1;
        history{n}=struct("L",L,"g",sv,"H",H,"R",Q,"d",correction,"lambda",lam,...
            "dual",dual,"slope",der);
    end
end

function out=make_result(R,F,G,l,s,t,face,count)
out=struct("R",R,"F",F,"G",G,"lambda",l,"s",s,"t",t,"kind",face,"calls",count);
end

function [x,kind]=support_column(z,c,j,t,m)
a=z.L.'*c;
b=z.H(j,:).';
g=z.g;
tol=m*eps*max(1,g(1));
rankM=nnz(g>tol);
k=numel(a)-rankM;
if k>=2
    A=a(rankM+1:end); B=b(rankM+1:end);
    h=t-a(1:rankM).'*b(1:rankM);
    radius=norm(A)*norm(B);
    if abs(h)<=radius+m*eps
        Q=eye(k);
        if radius==0
            Q(end,end)=z.d;
        else
            p=B/norm(B);
            f=min(1,max(-1,h/radius));
            eyeK=eye(k);
            [~,i]=min(abs(A));
            y=direction(A/norm(A),eyeK(:,i),f,sqrt(max(0,(1-f)*(1+f))));
            [F,~]=qr(p);
            [G,~]=qr(y);
            F(:,1)=p;
            G(:,1)=y;
            D=eye(k);
            D(end,end)=z.d*sign(det(G*F.'));
            Q=G*D*F.';
        end
        x=z.L*[b(1:rankM); Q*B];
        kind="null_face";
        return
    end
end
if z.d<0
    start=find(g-g(end)<=tol,1);
    if start<numel(g)
        A=a(start:end); B=b(start:end);
        [U,e]=eig((A*B.'+B*A.')/2,"vector");
        [v,order]=sort(e);
        eta=(a.'*b-t)/2;
        if eta>=v(1)-m*eps && eta<=v(end)+m*eps
            weight=0;
            if v(end)>v(1)
                weight=max(0,min(1,(eta-v(1))/(v(end)-v(1))));
            end
            y=sqrt(1-weight)*U(:,order(1))+sqrt(weight)*U(:,order(end));
            x=z.L*[b(1:start-1); B-2*y*(y.'*B)];
            kind="reflection_face";
            return
        end
    end
end
x=z.R(:,j);
kind="regular";
end

function R=completion(w,x,j,s)
[Q,~]=qr(x);
N=Q(:,2:end);
k=setdiff(1:numel(x),j,'stable');
base=zeros(numel(x));
base(:,j)=x;
base(:,k)=N;
[U,~,V]=svd(N(k,:).'.*w(k).');
U(:,end)=U(:,end)*s*sign(det(base))*sign(det(U*V.'));
R=base;
R(:,k)=N*(U*V.');
end

function x=direction(c,v,t,ss)
z=v-c*(c.'*v);
z=z-c*(c.'*z);
if norm(z)<=16*eps
    I=eye(numel(c));
    [~,k]=min(abs(c));
    z=I(:,k)-c*c(k);
end
x=t*c+ss*z/norm(z);
end
