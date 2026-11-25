function phase03_analyze(runDir,method)
% 使用原始區塊矩陣獨立求 SVD，並以每十個 JPEG 品質條件保存檢查點。
arguments
    runDir (1,1) string
    method (1,1) string
end
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'phase01','MATLAB'));
data=fullfile(runDir,'04_影像與矩陣');
out=fullfile(runDir,'05_統計分析');
base=fullfile(data,method);
common=fullfile(data,'common.h5');
embed=fullfile(base,'embedding.h5');
attacks=fullfile(base,'attacks.h5');
frozen=jsondecode(fileread(fullfile(runDir,'03_校準與凍結','frozen.json')));
c=frozen.methods.(method);
j=str2double(extractAfter(method,2));
left=extractBetween(method,2,2)=="U";
q=c.q_rad;
classification=string(c.classification);
n=12696;
expected=vec(common,"/original/expected");
target=vec(embed,"/geometry/target_phi");
valid=logical(vec(embed,"/geometry/embed_success"));
centers=vec(embed,"/geometry/target_index");
unbounded=vec(embed,"/geometry/unconstrained_index");
A=pages(common,"/original/matrices"); FA=factor(A);
B=pages(embed,"/B/matrices"); FB=factor(B);
B0=pages(embed,"/B0/matrices"); F0=factor(B0);
assert(size(A,3)==3*n && isequal(size(A),size(B),size(B0)));
[phiA,gA,classA]=featureData(FA,j,left);
[phiB,gB,classB]=featureData(FB,j,left);
[phi0,g0,class0]=featureData(F0,j,left);
ideal=vec(embed,"/geometry/theoretical_loss");
turn=vec(embed,"/geometry/actual_rotation_angle");
series=["A_to_B","B_to_B0","A_to_B0","B0_to_C","B_to_C","A_to_C"];
fReady=fullfile(base,'matlab_checks');
if ~isfolder(fReady), mkdir(fReady); end
for block=0:10
    piece=compose("part%02d",block);
    marker=fullfile(fReady,piece+".csv");
    if isfile(marker), continue; end
    partial=fullfile(fReady,piece+".pending.h5");
    final=fullfile(fReady,piece+".h5");
    if isfile(partial), delete(partial); end
    rows=cell(10,4);
    count=1;
    if block==0
        pairs={A,B,FA,FB;B,B0,FB,F0;A,B0,FA,F0};
        [X,fields,violation]=allPairs(pairs,phiA,phiB,phi0,phiB,phi0,phi0,j,left);
        writeData(partial,"/unattacked/pairs",X,fields,n);
        F=describe(method,classification,0,series(1:3),X,fields,n);
        [D,df,wrong,nearBoundary]=decision(phi0,phiA,phi0,g0,class0,q,centers,expected,valid);
        writeData(partial,"/unattacked/decision",D,df,n);
        Q=describe(method,classification,0,"decision",D,df,n);
        norms=reshape(sum(A.^2,[1 2]),[],1);
        e=max(abs(ideal-X(:,1,1))./max(1,norms));
        assert(e<=1e-10);
        compareSVD(common,"/original",FA,j,left,n);
        compareSVD(embed,"/B",FB,j,left,n);
        sError=compareSVD(embed,"/B0",F0,j,left,n);
        [mismatch,unexplained]=auditBits(embed,"/B0/block_bits",wrong,expected,nearBoundary,class0);
        orthogonal=~startsWith(method,"S");
        if orthogonal
            preservation=max(abs(FA.s-FB.s)./max(1,FA.s(:,1)),[],'all');
            assert(preservation<=1e-10);
        end
        fit=abs(phiB-target);
        fitEmbedded=fit; fitEmbedded(~valid)=NaN;
        diagnostics=[phiA phiB phi0 gA gB g0 classA classB class0 turn ideal double(valid) ...
            double(centers>=0 & centers~=unbounded) fit fitEmbedded double(~valid)];
        names=["phi_A","phi_B","phi_B0","gap_A","gap_B","gap_B0","class_A","class_B","class_B0", ...
            "rotation","theory_loss","embed_success","reselected","B_angle_target_error", ...
            "B_angle_target_error_success","embed_failure"];
        if startsWith(method,"G")
            diagnostics(:,end+1)=double(unbounded<0 & ~valid)+2*double(unbounded>=0 & ~valid);
            names(end+1)="givens_failure_code";
            diagnostics(:,end+1)=vec(embed,"/geometry/feasible_planes");
            names(end+1)="givens_feasible_planes";
        end
        for key=["orthogonality","primal","dual","scaled_lambda","svd_calls","solve_seconds_per_block_batch_average"]
            if startsWith(method,"P")
                diagnostics(:,end+1)=vec(embed,"/geometry/"+key); %#ok<AGROW>
                names(end+1)=key; %#ok<AGROW>
            elseif key=="orthogonality"
                diagnostics(:,end+1)=vec(fullfile(base,"geometry_audit.h5"),"/orthogonality");
                names(end+1)=key;
            end
        end
        if startsWith(method,"P")
            diagnostics(:,end+1)=diagnostics(:,names=="dual")-diagnostics(:,names=="primal");
            names(end+1)="certificate_gap";
            d=diagnostics(valid,names=="certificate_gap")./max(1,sum((FA.s(valid,:)./max(realmin,FA.s(valid,1))).^2,2));
            assert(all(d>=-1e-10 & d<=1e-8));
        end
        writeData(partial,"/unattacked/embedding",diagnostics,names,n);
        F=[F;describe(method,classification,0,"embedding",diagnostics,names,n)];
        T=checks(method,classification,0,F0,violation,sError,mismatch,unexplained,e);
        rows(1,:)={F,Q,T,relation(method,classification,0,gA,D,df,X(:,end,3),class0,n)};
        count=2;
    else
        if ~isfile(attacks), return; end
        for quality=(block-1)*10+(1:10)
            group="/jpeg/"+compose("QF%03d",quality);
            C=pages(attacks,group+"/matrices");
            FC=factor(C);
            [phiC,gC,classC]=featureData(FC,j,left);
            pairs={B0,C,F0,FC;B,C,FB,FC;A,C,FA,FC};
            [X,fields,violation]=allPairs(pairs,phi0,phiC,phiB,phiC,phiA,phiC,j,left);
            writeData(partial,group+"/pairs",X,fields,n);
            F=describe(method,classification,quality,series(4:6),X,fields,n);
            [D,df,wrong,nearBoundary]=decision(phiC,phiA,phi0,gC,classC,q,centers,expected,valid);
            writeData(partial,group+"/decision",D,df,n);
            Q=describe(method,classification,quality,"decision",D,df,n);
            [mismatch,unexplained]=auditBits(attacks,group+"/block_bits",wrong,expected,nearBoundary,classC);
            sError=compareSVD(attacks,group,FC,j,left,n);
            cross=2*reshape(sum((B0-A).*(C-B0),[1 2]),[],1);
            a0=reshape(sum((B0-A).^2,[1 2]),[],1);
            e=max(abs(X(:,1,3)-a0-X(:,1,1)-cross)./max(1,a0+X(:,1,1)+abs(cross)));
            assert(e<=128*eps);
            T=checks(method,classification,quality,FC,violation,sError,mismatch,unexplained,e);
            rows(count,:)={F,Q,T,relation(method,classification,quality,g0,D,df,X(:,end,1),classC,n)};
            count=count+1;
        end
    end
    rows=rows(1:count-1,:);
    for k=1:4
        names=["matrix","decision","checks","relation"];
        path=fullfile(fReady,piece+"_"+names(k)+".csv");
        writetable(vertcat(rows{:,k}),path);
    end
    movefile(partial,final,'f');
    completed=count-1;
    writetable(table(completed),marker);
    fprintf('%s MATLAB %d/100\n',method,block*10);
end
for key=["matrix","decision","checks","relation"]
    rows=cell(11,1);
    for k=0:10
        rows{k+1}=readtable(fullfile(fReady,compose("part%02d_",k)+key+".csv"),'TextType','string');
    end
    writetable(vertcat(rows{:}),fullfile(out,method+"_"+key+"_matlab.csv"));
end
end

function x=pages(file,path)
x=reshape(permute(h5read(file,path),[2 1 3 4]),4,4,[]);
end

function x=vec(file,path)
x=h5read(file,path);
if ~isnumeric(x) && ~islogical(x), x=strcmpi(string(x),"true"); end
x=double(x(:));
end

function [phi,g,tag]=featureData(F,j,left)
if left, X=F.U; else, X=F.V; end
x=reshape(X(:,j,:),4,[]);
p=sum(x,1)/2;
phi=atan2(vecnorm(x-p/2),abs(p)).';
a=abs(F.s-F.s(:,j)); a(:,j)=Inf;
g=min(a,[],2);
t=32*eps*F.s(:,1);
zero=F.s(:,j)<=t;
tag=ones(size(g));
tag(g<=1e-6*F.s(:,1))=2;
tag(g<=t)=3;
tag(zero & g>t)=4;
tag(zero & g<=t)=5;
end

function t=projective(P,Q)
d=sum(P.*Q,1);
y=(d>=0)-(d<0);
sb=sqrt(sum((Q-y.*P).^2,1));
sa=sqrt(sum((Q+y.*P).^2,1));
t=reshape(2*atan2(sb,sa),4,[]).';
end

function [X,names,violation]=allPairs(pairs,p1,p2,p3,p4,p5,p6,j,left)
phi={p1,p2;p3,p4;p5,p6};
count=size(pairs{1,1},3);
names=["D_F","D_sigma","R","nuclear_delta","delta_sigma_"+(1:4), ...
    "theta_U_"+(1:4),"theta_V_"+(1:4),"feature_shift","selected_projective_angle"];
X=zeros(count,numel(names),3);
violation=0;
for k=1:3
    A=pairs{k,1}; B=pairs{k,2}; F=pairs{k,3}; G=pairs{k,4};
    E=B-A;
    DF=reshape(sum(E.^2,[1 2]),[],1);
    independent=reshape(pagenorm(E,'fro'),[],1).^2;
    assert(max(abs(DF-independent)./max(1,DF))<=32*eps);
    delta=G.s-F.s;
    DS=sum(delta.^2,2);
    rest=DF-DS;
    energy=reshape(sum(A.^2+B.^2,[1 2]),[],1);
    assert(max(max(0,-rest)./max(1,energy))<=128*eps);
    u=projective(F.U,G.U); v=projective(F.V,G.V);
    if left, theta=u(:,j); else, theta=v(:,j); end
    shift=phi{k,2}-phi{k,1};
    violation=max(violation,max(max(0,abs(shift)-theta)));
    assert(violation<=128*eps);
    X(:,:,k)=[DF DS rest sum(delta,2) delta u v shift theta];
end
end

function [D,names,wrong,nearBoundary]=decision(phi,initial,clean,g,tag,q,k,expected,ok)
U=floor((pi/2)/q);
if U*q>pi/2, U=U-1; end
a=min(U,max(0,floor(phi/q)));
b=min(U,a+1);
idx=b;
closer=abs(phi-a*q)<abs(phi-b*q);
idx(closer)=a(closer);
bits=mod(idx,2);
wrong=bits~=expected;
e=phi-k*q;
below=phi-(k-.5)*q; below(k<=0)=inf;
above=(k+.5)*q-phi; above(k>=U)=inf;
margin=min(below,above);
e(~ok)=NaN; margin(~ok | isinf(margin))=NaN;
nearBoundary=abs(phi-(min(a,b)+.5)*q)<=128*eps;
names=["phi","offset","abs_offset","normalized_offset","target_margin","feature_delta_from_A","feature_delta_from_B0", ...
    "nearest_gap","collision_distance","rank_class","decoded_index","decision","block_error", ...
    "inside_sufficient_radius"];
D=[phi e abs(e) e/q margin phi-initial phi-clean g g/sqrt(2) tag idx bits double(wrong) ...
    double(ok & abs(e)<q/2)];
assert(~any(D(:,end)>0 & wrong));
end

function [m,u]=auditBits(file,path,wrong,expected,nearBoundary,tag)
other=vec(file,path)~=expected;
diff=other~=wrong;
m=nnz(diff);
% 相等奇異值的基底與數值判決邊界可使跨平台判決相異，保留計數。
u=nnz(diff & ~nearBoundary & tag~=3 & tag~=5);
end

function error=compareSVD(file,path,F,j,left,n)
s=reshape(h5read(file,path+"/sigma"),4,3*n).';
error=max(abs(F.s-s)./max(1,F.s(:,1)),[],'all');
assert(error<=128*eps);
if left, key="/U"; X=F.U; else, key="/V"; X=F.V; end
Y=pages(file,path+key);
a=projective(X,Y);
d=abs(F.s-F.s(:,j)); d(:,j)=Inf;
g=min(d,[],2);
resolvable=g>1e-6*F.s(:,1);
assert(all(a(resolvable,j)<=2048*eps*F.s(resolvable,1)./g(resolvable)));
end

function T=describe(method,classification,quality,pairs,X,names,n)
scopes=["Y","U","V","All"];
p=[.1;.25;.5;.75;.9;.95];
M=numel(names);
R=cell(numel(pairs)*4,1);
for k=1:numel(pairs)
    for c=1:4
        pos=1:(3*n);
        if c~=4, pos=(c-1)*n+(1:n); end
        y=X(pos,:,k);
        z=sort(y,1,'MissingPlacement','last');
        count=sum(isfinite(y),1);
        N=max(1,count);
        offsets=(0:M-1)*size(y,1);
        h=1+(N-1).*p;
        a=floor(h); b=min(ceil(h),N);
        values=z(a+offsets)+(h-a).*(z(b+offsets)-z(a+offsets));
        d=[count;mean(y,1,'omitnan');std(y,1,1,'omitnan');z(1,:);z(N+offsets);values].';
        t=array2table(d,'VariableNames',{'count','mean','std','min','max','p10','p25','median','p75','p90','p95'});
        R{(k-1)*4+c}=addvars(t,repmat(method,M,1),repmat(classification,M,1),repmat(quality,M,1),...
            repmat(scopes(c),M,1),repmat(pairs(k),M,1),names(:),'Before',1,...
            'NewVariableNames',{'method','classification','qf','scope','comparison','metric'});
    end
end
T=vertcat(R{:});
end

function T=checks(method,classification,qf,F,angle_violation,sigma_error,bit_disagreement,...
    unexplained_bit_disagreement,formula_error)
reconstruction=F.reconstruction; orthogonality=F.orthogonality;
T=table(method,classification,qf,reconstruction,orthogonality,angle_violation,sigma_error, ...
    bit_disagreement,unexplained_bit_disagreement,formula_error);
end

function T=relation(method,classification,qf,g,D,fields,theta,tag,n)
scopes=["Y","U","V","All"];
rows=cell(4,6);
err=D(:,fields=="block_error");
for c=1:4
    x=1:3*n;
    if c<4, x=(c-1)*n+(1:n); end
    for r=0:5
        pos=x;
        if r>0, pos=x(tag(x)==r); end
        count=numel(pos);
        rho=NaN;
        if count>1 && std(g(pos))>0 && std(theta(pos))>0
            r0=corrcoef(g(pos),theta(pos));
            rho=r0(1,2);
        end
        rows{c,r+1}=table(method,classification,qf,scopes(c),r,count,rho,mean(theta(pos)),mean(err(pos)), ...
            'VariableNames',{'method','classification','qf','scope','rank_class','count', ...
            'pearson_gap_angle','mean_angle','block_ber'});
    end
end
T=vertcat(rows{:});
end

function writeData(file,path,X,fields,n)
V=permute(reshape(X, n,3,size(X,2),size(X,3)),[3 1 2 4]);
% 強制四維，MATLAB 會省略尾端為一的維度。
shape=[numel(fields),n,3,size(X,3)];
h5create(file,path,shape,'Datatype','double','ChunkSize',[numel(fields),512,1,1],'Deflate',1,'Shuffle',true);
h5write(file,path,V);
h5writeatt(file,path,'axes','comparison,channel,block,field');
h5writeatt(file,path,'fields',char(strjoin(fields,",")));
end
