function validate_allocations(runDir)
% 用 MATLAB 頁面 SVD 重算正式 A，並獨立求最近合法中心及投影。
arguments
    runDir (1,1) string
end
common=fullfile(runDir,'05_原始矩陣','common.h5');
raw=h5read(common,'/original/A');
A=reshape(permute(raw,[2 1 3 4]),4,4,[]);
F=factor(A);
bits=reshape(double(h5read(common,'/payload/expected_blocks')),[],1);
methodNames=["G0","S1","S2","S3","S4","S5","S6","S7"];
config=jsondecode(fileread(fullfile(runDir,'01_實驗配置','frozen_parameters.json')));
checks = struct();
% 包含零、秩一、重根、近秩虧，以及 S1 負分配的人工案例。
toy = [0 0 0 0; 9 0 0 0; 8 4 4 2; 8 1 1e-10 0; 9 1 .2 .01; 5 3 1 1];
toyBit=[0;1;1;0;0;1]; 
for k=2:8
    [TX,kt]=model(toy,toyBit,5,k-1);
    assert(all(TX>=0,'all') && all(abs(sum(TX,2)-5*kt)<1e-12));
    assert(all(mod(kt,2)==toyBit));
end
for k=1:8
    m=methodNames(k);
    file=fullfile(runDir,'06_嵌入與擾動數據',m+"_embed.h5");
    raw=h5read(file,'/B/A');
    B=reshape(permute(raw,[2 1 3 4]),4,4,[]);
    s=F.s;
    xPy=columns(file,'/embedding/s_modified');
    sPy=columns(file,'/embedding/s_original');
    step=config.methods.(m).parameter;
    componentError=max(abs(s(:)-sPy(:))./max(1,abs(s(:))));
    df=reshape(sum((B-A).^2,[1 2]),[],1);
    local=sum((xPy-sPy).^2,2);
    frobenius=max(abs(df-local)./max(1,df));
    if k==1
        assert(frobenius < 1e-3 && componentError < 3e-5);
        modelError=NaN; kkt=NaN; sameCenter=NaN; fixedFormulaError=NaN;
    else
        [x,khat,kkt]=model(sPy,bits,step,k-1);
        [xNative,kiNative]=model(s,bits,step,k-1);
        tgt=reshape(h5read(file,'/embedding/T'),[],1);
        expected=reshape(h5read(file,'/embedding/k'),[],1);
        assert(all(khat == expected));
        assert(all(abs(sum(x,2)-tgt)<=128*eps*max(1,tgt)));
        modelError=max(abs(x-xPy),[],'all');
        sameCenter=nnz(kiNative~=expected);
        if sameCenter>0
            residual=abs((tgt-sum(sPy,2)));
            boundary=abs(residual-step);
            feasibleBoundary=abs(tgt-reshape(h5read(file,'/embedding/T_min'),[],1));
            assert(all(min(boundary(kiNative~=expected),feasibleBoundary(kiNative~=expected)) ...
                <=256*eps*max(1,tgt(kiNative~=expected))));
        end
        y=reshape(h5read(file,'/embedding/y'),[],1);
        projected=h5read(file,'/embedding/projected');
        if iscell(projected) || iscategorical(projected) || isstring(projected)
            projected=strcmpi(string(projected),'true');
        end
        projected=logical(projected(:));
        if k==2
            predicted=y.^2/4;
        elseif k<=6
            predicted=y.^2;
        else
            total=sum(sPy,2);
            coeff=sum(sPy.^2,2)./total.^2;
            coeff(total==0)=1/4;
            predicted=y.^2.*coeff;
        end
        fixedFormulaError=max(abs(predicted(~projected)-local(~projected))./max(1,local(~projected)));
        assert(modelError<1e-9 && frobenius<1e-9 && kkt<1e-9);
        assert(fixedFormulaError<1e-9);
        assert(componentError<1e-9);
        assert(all(abs(xNative(kiNative==expected,:)-xPy(kiNative==expected,:))<1e-9,'all'));
    end
    checks.(m)=struct('native_sigma_max_scaled',componentError,'frobenius_identity_max_scaled',frobenius, ...
        'allocation_max_absolute',modelError,'projection_kkt_max_absolute',kkt, ...
        'fixed_weight_loss_formula_error',fixedFormulaError,'native_centers_at_numeric_boundary',sameCenter);
    fprintf('%s MATLAB independent model passed\n',m);
end
environment=struct('matlab',version,'release',version('-release'),'products',ver,'passed',true,'model_checks',checks);
fid=fopen(fullfile(runDir,'09_Python驗證','matlab_allocation_gate.json'),'w','n','UTF-8');
assert(fid~=-1)
fprintf(fid,'%s\n',jsonencode(environment,'PrettyPrint',true));
fclose(fid);
end

function x=columns(file,path)
x=reshape(h5read(file,path),4,[]).';
end

function [x,k,kkt]=model(s,bit,q,m)
S=sum(s,2);
if m==1
    w=ones(size(s))/4;
elseif m<=5
    w=zeros(size(s)); w(:,m-1)=1;
else
    w=s./S;
    if m==7, w=fliplr(w); end
    w(S==0,:)=1/4;
end
lower=bit+2*max(0,floor((S/q-bit)/2));
higher=lower+2;
nearest=lower;
greater=abs(higher*q-S)<=abs(lower*q-S);
nearest(greater)=higher(greater);
tol=64*eps*max(max(max(1,S),q),nearest*q);
k=nearest;
if m>1
    r=s./w; r(w==0)=Inf;
    minT=max(0,S-min(r,[],2));
    required=bit+2*max(0,ceil(((minT-tol)/q-bit)/2));
    k=max(nearest,required);
end
T=k*q;
x=s+(T-S).*w;
kkt=0;
if m==1
    % 單純形活動集合以獨立的單調函數二分求根。
    ids=find(min(x,[],2)<0);
    z=s(ids,:);
    target=T(ids);
    L=-max(z,[],2); R=target-min(z,[],2);
    for iter=1:60
        h=(L+R)/2;
        tooHigh=sum(max(z+h,0),2)>target;
        R(tooHigh)=h(tooHigh);
        L(~tooHigh)=h(~tooHigh);
    end
    tau=(L+R)/2;
    x(ids,:)=max(z+tau,0);
    x(T==0,:)=0;
    residual=abs(sum(x(ids,:),2)-target);
    comp=abs(x(ids,:).*max(0,-(z+tau)));
    kkt=max([0; residual(:); comp(:)]);
end
assert(all(x>=-tol,'all'))
x=max(0,x);
end
