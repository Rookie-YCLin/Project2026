function phase03_math_validation(out)
% 從獨立樣本集中核對三類公式、四分支證書以及重建矩陣。
tic
packet=jsondecode(fileread(fullfile(out,'cases.json')));
cases=packet.cases;
if isstruct(cases), cases=num2cell(cases); end
items=cell(numel(cases),1);
bad={};
n=0;
keys=["orthogonality","projection","angle","determinant","sigma","primal_dual_gap","frobenius"];
for i=1:numel(cases)
    a=cases{i};
    try
        s=a.sigma(:);
        if max(s)>0, s=s/max(s); end
        w=s.^2;
        c=a.c(:);
        j=a.j+1;
        phi=a.phi;
        scale=max(1,sum(w));
        I=eye(4);
        getLoss=@(M) sum((M-I).^2.*w.','all');
        [S,G,R,planes,kG,thetaG,thetaR]=phase03_simple_candidates(w,c,j,phi);
        P=phase03_procrustes(w,c,j,phi);
        [~,k]=max([P.F]);
        chosen=P(k);
        dS=getLoss(S); dR=getLoss(R); dP=getLoss(chosen.R);
        dG=NaN;
        V=[{R};{P.R}.'];
        roots=[{S}; V];
        if ~isempty(G), dG=getLoss(G); roots{end+1}=G; V{end+1}=G; end
        angle=0; ort=0; detErr=0; projection=0; Frob=0;
        for l=1:numel(roots)
            y=roots{l}(:,j);
            d=c.'*y;
            angle=max(angle,abs(atan2(norm(y-c*d),abs(d))-phi));
        end
        for l=1:numel(V)
            X=V{l};
            ort=max(ort,norm(X.'*X-I,'fro'));
        end
        for l=1:numel(P)
            X=P(l).R;
            detErr=max(detErr,abs(det(X)-P(l).s));
            projection=max(projection,abs(c.'*X(:,j)-P(l).t));
            Frob=max(Frob,abs(getLoss(X)-(2*sum(w)-2*P(l).F))/scale);
        end
        Frob=max(Frob,abs(dS-4*w(j)*sin(thetaR/2)^2)/scale);
        aux=R(:,j); aux(j)=0;
        if norm(aux)>0, aux=aux/norm(aux); end
        Frob=max(Frob,abs(dR-4*sin(thetaR/2)^2*(w(j)+w.'*aux.^2))/scale);
        if ~isempty(G)
            Frob=max(Frob,abs(dG-4*(w(j)+w(kG))*sin(thetaG/2)^2)/scale);
        end
        if strcmp(a.side,"U")
            A=a.basis*diag(s)*a.other.';
            B=a.basis*chosen.R*diag(s)*a.other.';
        else
            A=a.other*diag(s)*a.basis.';
            B=a.other*diag(s)*chosen.R.'*a.basis.';
        end
        Frob=max(Frob,abs(norm(B-A,'fro')^2-dP)/scale);
        dynamicError=0;
        for M=V.'
            X=M{1};
            if strcmp(a.side,"U")
                B=a.basis*X*diag(s)*a.other.';
            else
                B=a.other*diag(s)*X.'*a.basis.';
            end
            dynamicError=max(dynamicError,max(abs(svd(B)-s)));
        end
        lower=max(0,max([P.F]-[P.G]))/scale;
        inequality=max([0,dS-dP,dP-dR,dP-dG],[],"omitnan")/scale;
        err=[ort projection angle detErr dynamicError max(0,max([P.G]-[P.F]))/scale Frob];
        row=struct("id",a.id,"category",string(a.category),"side",string(a.side),"j",a.j,"phi",phi,...
            "D_S",dS,"D_G",dG,"D_R",dR,"D_P",dP,"planes",planes,...
            "dual_below_primal",lower,"inequality_error",inequality,"p_kind",chosen.kind,...
            "det_branch",chosen.s,"projected_value",chosen.t,"lamb_scaled",chosen.lambda,...
            "all_svd_calls",sum([P.calls]));
        for j1=1:numel(keys), row.(keys(j1))=err(j1); end
        for j1=1:4
            row.("F"+j1)=P(j1).F; row.("G"+j1)=P(j1).G; row.("branch"+j1)=P(j1).kind;
        end
        n=n+1; items{n}=row;
        thresh=struct2array_order(packet.limits,keys);
        if any(err>thresh) || lower>1e-10 || inequality>1e-10
            bad{end+1}=struct("case",a,"row",row); %#ok<AGROW>
        end
    catch ME
        bad{end+1}=struct("case",a,"message",getReport(ME,'extended','hyperlinks','off')); %#ok<AGROW>
    end
    if mod(i,200)==0, fprintf('MATLAB %d failures %d\n',i,numel(bad)); end
end
if n>0, writetable(struct2table(vertcat(items{1:n})),fullfile(out,'matlab_cases.csv')); end
write_json(out,"matlab_failures.json",bad);
assert(isempty(bad),"Phase 03 mathematical gate failed");
f=phase03_procrustes([16;9;4;1]/16,[1;0;0;0],1,acos(.5));
assert(abs(16*f(1).F-21.5)<1e-10 && abs(16*f(1).G-21.5)<1e-9);
assert(abs(16*f(1).lambda+17)<1e-8);
T=struct2table(vertcat(items{1:n}));
mx=struct();
for j=1:numel(keys), mx.(keys(j))=max(T.(keys(j))); end
v=struct("passed",true,"case_count",n,"elapsed_seconds",toc,"matlab",version,...
    "maxima",mx,"example",struct("ordinary_dual",22,"det_positive_primal",16*f(1).F,...
    "det_positive_dual",16*f(1).G,"lambda_sample",16*f(1).lambda,"squared_loss",60-32*f(1).F));
write_json(out,"matlab_summary.json",v);
end

function out=struct2array_order(s,keys)
out=zeros(size(keys));
for i=1:numel(keys), out(i)=s.(keys(i)); end
end

function write_json(out,name,v)
f=fopen(fullfile(out,name),'w','n','UTF-8');
assert(f~=-1);
fprintf(f,'%s\n',jsonencode(v,'PrettyPrint',true));
fclose(f);
end
