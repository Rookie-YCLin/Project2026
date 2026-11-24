function analyze_method(runDir,method)
% 分方法讀取正式矩陣並獨立計算 MATLAB SVD、逐塊指標與描述統計。
arguments
    runDir (1,1) string
    method (1,1) string
end
inDir = fullfile(runDir,'06_嵌入與擾動數據');
file = fullfile(inDir,method+"_embed.h5");
jpeg = fullfile(inDir,method+"_jpeg.h5");
output = fullfile(inDir,method+"_matlab.h5");
assert(~isfile(output));
raw = fullfile(runDir,'05_原始矩陣','common.h5');
frozen = jsondecode(fileread(fullfile(runDir,'01_實驗配置','frozen_parameters.json')));
parameter = frozen.methods.(method).parameter;
expected = double(vec(raw,"/payload/expected_blocks"));
A = pages(raw,"/original/A"); B = pages(file,"/B/A"); B0 = pages(file,"/B0/A");
n = size(A,3)/3;
F = factor(A); G = factor(B); H = factor(B0);
if method=="G0"
    % A 的儲存位於 G0 結果，各組共用該原始分解。
    saveFactors(output,"/original",F,n);
end
saveFactors(output,"/B",G,n); saveFactors(output,"/B0",H,n);
allStats = cell(3+2*100,1); jj=0;
allSpectral=cell(103,1);
allSpectral{1}=saveSpectrum("A",F,n);
allSpectral{2}=saveSpectrum("B",G,n);
allSpectral{3}=saveSpectrum("B0",H,n);
starts = {"A_to_B",A,B,F,G; "B_to_B0",B,B0,G,H; "A_to_B0",A,B0,F,H};
for k=1:size(starts,1)
    [values,fields,check] = pair(starts{k,2},starts{k,3},starts{k,4},starts{k,5});
    saveColumns(output,"/unattacked/"+starts{k,1},values,fields,n);
    jj=jj+1;
    allStats{jj}=summarize("unattacked",starts{k,1},values,fields,n);
    assert(check<=128*eps);
end
diagNames = ["nonnegative_trigger","projected","reselected","rank_reversal","adjacent_tie","zero_sum_fallback","y"];
if method=="G0"
    diagNames = ["rank_reversal","adjacent_tie"];
end
embed = zeros(3*n,numel(diagNames));
for k=1:numel(diagNames), embed(:,k)=vec(file,"/embedding/"+diagNames(k)); end
diagStats=summarize("unattacked","allocation",embed,diagNames,n);
if method=="G0"
    target = reshape(h5read(file,'/embedding/T_components'),2,[]).';
else
    target = vec(file,"/embedding/T");
end
qStats=cell(101,1); qAssociations=cell(101,1);
[values,fields,errorFlag]=qim(H.s,H.s,method,parameter,target,expected);
saveColumns(output,"/unattacked/decision",values,fields,n);
qStats{1}=summarize("unattacked","decision",values,fields,n);
qAssociations{1}=associations("unattacked",values,errorFlag,n);
mismatch=nnz(errorFlag~= (double(vec(file,"/unattacked/block_hard"))~=expected));
checks=zeros(100,7);
for q=1:100
    label=compose("QF%03d",q);
    group="/jpeg/"+label;
    C=pages(jpeg,group+"/A");
    J=factor(C);
    saveFactors(output,group,J,n);
    allSpectral{q+3}=saveSpectrum(label,J,n);
    [local,names,mirsky1]=pair(B0,C,H,J);
    [total,~,mirsky2]=pair(A,C,F,J);
    saveColumns(output,group+"/B0_to_C",local,names,n);
    saveColumns(output,group+"/A_to_C",total,names,n);
    jj=jj+1;
    allStats{jj}=summarize(label,"B0_to_C",local,names,n);
    jj=jj+1;
    allStats{jj}=summarize(label,"A_to_C",total,names,n);
    [values,fields,errorFlag]=qim(J.s,H.s,method,parameter,target,expected);
    saveColumns(output,group+"/decision",values,fields,n);
    qStats{q+1}=summarize(label,"decision",values,fields,n);
    qAssociations{q+1}=associations(label,values,errorFlag,n);
    wrong=double(vec(jpeg,group+"/block_hard"))~=expected;
    ps=reshape(h5read(jpeg,group+"/s_extracted"),4,[]).';
    epsFactor=128*eps;
    if method=="G0", epsFactor=8*double(eps('single')); end
    % Weyl 與 SVD 的後向誤差以整個區塊的矩陣尺度為準。
    svError=max(abs(J.s-double(ps))./max(1,J.s(:,1)),[],'all');
    assert(svError<=epsFactor);
    du=reshape(sum((C-A).^2,[1 2]),[],1);
    % 總擾動的兩步交叉項。
    cross=2*reshape(sum((B0-A).*(C-B0),[1 2]),[],1);
    df0=reshape(sum((B0-A).^2,[1 2]),[],1);
    crossError=max(abs(du-(df0+local(:,1)+cross))./max(1,df0+local(:,1)+abs(cross)));
    assert(crossError<=128*eps);
    checks(q,:)=[q,J.reconstruction,J.orthogonality,max(mirsky1,mirsky2),svError,nnz(wrong~=errorFlag),crossError];
    if mod(q,10)==0, fprintf('%s MATLAB %d/100\n',method,q); end
end
path=fullfile(runDir,'07_統計分析');
writetable(vertcat(allStats{:}),fullfile(path,method+"_matrix_stats.csv"));
writetable(vertcat(allSpectral{:}),fullfile(path,method+"_spectral_stats.csv"));
writetable(diagStats,fullfile(path,method+"_allocation_stats.csv"));
writetable(vertcat(qStats{:}),fullfile(path,method+"_decision_stats.csv"));
writetable(vertcat(qAssociations{:}),fullfile(path,method+"_association.csv"));
T=array2table(checks,'VariableNames',{'qf','matlab_reconstruction','matlab_orthogonality', ...
    'mirsky_error_scaled','sigma_python_difference_scaled','block_bit_mismatches','cross_term_error_scaled'});
T.png_mismatches=repmat(mismatch,100,1);
writetable(T,fullfile(runDir,'10_交叉驗證',method+"_matlab_checks.csv"));
    function T=saveSpectrum(label,K,num)
        d=-diff(K.s,1,2);
        g=[d(:,1),min(d(:,1:2),[],2),min(d(:,2:3),[],2),d(:,3)];
        scale=K.s(:,1);
        names=["gap12","gap23","gap34", "nearest_"+(1:4), "normalized_"+(1:4), ...
            "repeated_"+(1:4), "near_repeated_"+(1:4), "zero_singular_"+(1:4), "near_rank_deficient","zero_matrix"];
        data=[d,g,g./scale,g<=32*eps*scale,g<=1e-6*scale,K.s<=32*eps*scale,...
            K.s(:,4)<=1e-6*scale,scale==0];
        if label~="A" || method=="G0"
            h="/"+label;
            if startsWith(label,"QF"), h="/jpeg/"+label; end
            saveColumns(output,h+"/spectral",data,names,num);
        end
        T=summarize(label,"spectral",data,names,num);
    end
end

function A=pages(path,k)
A=reshape(permute(h5read(path,k),[2 1 3 4]),4,4,[]);
end

function x=vec(file,path)
x=h5read(file,path);
if iscell(x) || iscategorical(x) || isstring(x)
    x=strcmpi(string(x),'true');
end
x=x(:);
end

function [X,names,mirsky]=pair(A,B,F,G)
E=B-A;
df=reshape(sum(E.^2,[1 2]),[],1);
dfIndependent=reshape(pagenorm(E,'fro'),[],1).^2;
assert(max(abs(df-dfIndependent)./max(1,df))<=32*eps);
ds=G.s-F.s;
dsv=sum(ds.^2,2);
R=df-dsv;
nuclear=sum(G.s,2)-sum(F.s,2);
energy=reshape(sum(A.^2+B.^2,[1 2]),[],1);
mirsky=max(max(0,-R)./max(1,energy));
assert(mirsky<=128*eps);
X=[df dsv R nuclear abs(nuclear) ds abs(ds) angles_svd(F.U,G.U) angles_svd(F.V,G.V)];
names=["D_F","D_sigma","R","nuclear_delta","nuclear_abs","delta_"+(1:4),"abs_delta_"+(1:4), ...
       "theta_u_"+(1:4),"theta_v_"+(1:4)];
end

function [X,names,errorFlag]=qim(s,s0,method,p,target,expected)
if method=="G0"
    d=double(single([36*p,20*p]));
    S=double(single(s(:,1:2)));
    R=double(single(rem(S,d)));
    r0=double(single(rem(double(single(s0(:,1:2))),d)));
    e=S-target;
    delta=S-double(single(s0(:,1:2)));
    e0=double(single(s0(:,1:2)))-target;
    decision=R>d/2;
    nearest=min(cat(3,R,abs(R-d/2),d-R),[],3);
    X=[e,abs(e),delta,e0,nearest,double(R>d/2),double(r0>d/2), ...
        (3*double(decision(:,1))+double(decision(:,2)))/4];
    errorFlag=decision(:,1)~=expected;
    names=["e_s1","e_s2","abs_e_s1","abs_e_s2","jpeg_delta_s1","jpeg_delta_s2","e0_s1","e0_s2",...
        "nearest_boundary_s1","nearest_boundary_s2","decision_s1","decision_s2", ...
        "unattacked_decision_s1","unattacked_decision_s2","combined_score"];
else
    S=sum(s,2); S0=sum(s0,2);
    e=S-target; e0=S0-target; delta=S-S0;
    k=floor(S/p+0.5);
    offset=k-round(target/p);
    decision=mod(k,2);
    errorFlag=decision~=expected;
    X=[e,abs(e),e0,delta,e/p,p/2-abs(e),max(0,p/2-abs(S-k*p)),offset,mod(offset,2),decision,double(errorFlag)];
    names=["e","abs_e","e0","jpeg_nuclear_delta","z","target_margin","nearest_boundary","index_offset",...
        "offset_parity","decision","block_error"];
end
end

function C=associations(condition,X,err,n)
scopes=["Y";"U";"V";"All"];
cols=nan(4,3);
for c=1:4
    idx=1:3*n;
    if c<4, idx=(c-1)*n+(1:n); end
    k=double(err(idx));
    cols(c,1:2)=[numel(k),mean(k)];
    this=X(idx,2);
    if max(this)>min(this) && max(k)>min(k)
        r=corrcoef(this,k);
        cols(c,3)=r(1,2);
    end
end
C=table(repmat(condition,4,1),scopes,cols(:,1),cols(:,2),cols(:,3), ...
    'VariableNames',{'condition','scope','count','matlab_block_ber','correlation_col2_vs_block_error'});
end

function T=summarize(condition,pairName,X,fields,n)
scopes=["Y","U","V","All"];
p=[.5,.10,.25,.75,.90,.95]';
d=zeros(numel(fields)*4,11);
for c=1:4
    idx=1:3*n;
    if c<4, idx=(c-1)*n+(1:n); end
    x=X(idx,:);
    sorted=sort(x,1,"ascend","MissingPlacement","last");
    count=sum(isfinite(x),1);
    xx=x; xx(~isfinite(x))=0;
    avg=sum(xx,1)./count;
    sub=xx-avg; sub(~isfinite(x))=0;
    sd=sqrt(sum(sub.^2,1)./count);
    h=1+(max(count,1)-1).*p;
    a=floor(h); b=min(a+1,max(count,1));
    offset=(0:size(x,2)-1)*size(x,1);
    frac=h-a;
    quantiles=sorted(a+offset)+frac.*(sorted(b+offset)-sorted(a+offset));
    mat=[count;avg;quantiles(1,:);sd;sorted(1,:);sorted(max(count,1)+offset);quantiles(2:end,:)].';
    mat(count==0,2:end)=NaN;
    d((c-1)*numel(fields)+(1:numel(fields)),:)=mat;
end
T=array2table(d,'VariableNames',{'count','mean','median','std','min','max','p10','p25','p75','p90','p95'});
T=addvars(T,repmat(condition,height(T),1),repmat(pairName,height(T),1), ...
    repelem(scopes.',numel(fields)),repmat(fields(:),4,1),'Before',1, ...
    'NewVariableNames',{'condition','comparison','scope','metric'});
end

function saveFactors(file,path,F,n)
write(file,path+"/U",permute(reshape(F.U,4,4,n,3),[2 1 3 4]),"channel,block,row,singular_index");
write(file,path+"/sigma",reshape(F.s.',4,n,3),"channel,block,singular_index");
write(file,path+"/V",permute(reshape(F.V,4,4,n,3),[2 1 3 4]),"channel,block,row,singular_index");
end

function saveColumns(file,path,v,names,n)
write(file,path,reshape(v.',numel(names),n,3),"channel,block,field");
h5writeatt(file,path,'fields',char(strjoin(names,",")));
end

function write(file,path,v,axes)
s=size(v);
chunk=s;
chunk(end-1)=min(s(end-1),1024);
chunk(end)=1;
h5create(file,path,s,'Datatype','double','ChunkSize',chunk,'Deflate',1,'Shuffle',true);
h5write(file,path,v);
h5writeatt(file,path,'axes',char(axes));
h5writeatt(file,path,'dtype','double');
h5writeatt(file,path,'indices','HDF5 C-order, zero-based; channels Y,U,V');
end
