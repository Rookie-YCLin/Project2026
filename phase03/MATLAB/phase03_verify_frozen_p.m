function phase03_verify_frozen_p(runDir)
% 以凍結的實際目標重算 P 的四個分支，核對批次流程的可行目標。
arguments
    runDir (1,1) string
end
data=fullfile(runDir,'04_影像與矩陣');
out=fullfile(runDir,'01_數學驗證');
source=fullfile(data,'common.h5');
sigma=reshape(h5read(source,'/original/sigma'),4,[]);
n=size(sigma,2)/3;
records=cell(8,1);
m=0;
for side=["U","V"]
    factor=reshape(permute(h5read(source,"/original/"+side),[2 1 3 4]),4,4,[]);
    for j=1:4
        label="P"+side+j;
        f=fullfile(data,label,'embedding.h5');
        target=reshape(h5read(f,'/geometry/target_phi'),[],1);
        index=reshape(h5read(f,'/geometry/target_index'),[],1);
        recorded=reshape(h5read(f,'/geometry/branch_primal'),4,[]);
        bounds=reshape(h5read(f,'/geometry/branch_dual'),4,[]);
        neighbors=abs(sigma-sigma(j,:)); neighbors(j,:)=Inf;
        gap=min(neighbors,[],1)./max(realmin,sigma(1,:));
        sample=[];
        for c=1:3
            pos=(c-1)*n+(1:n);
            pos=pos(index(pos)>=0);
            [~,small]=sort(gap(pos),"ascend");
            sample=[sample, pos(small(1:8)),pos(round(linspace(1,numel(pos),16)))]; %#ok<AGROW>
        end
        sample=unique(sample);
        scores=zeros(numel(sample),5);
        for k=1:numel(sample)
            z=sample(k);
            w=(sigma(:,z)/max(realmin,sigma(1,z))).^2;
            c=factor(:,:,z).'*[.5;.5;.5;.5];
            c=c/norm(c);
            phi=target(z);
            b=phase03_procrustes(w,c,j,phi);
            independent=[b.F].';
            den=max(1,sum(w));
            err=max(abs(independent-recorded(:,z)))/den;
            mtGap=max([b.G]-[b.F])/den;
            pyGap=max(bounds(:,z)-recorded(:,z))/den;
            correction=max(abs([b.t]-[1,1,-1,-1]*cos(phi)));
            constraint=zeros(4,1);
            for branch=1:4
                x=b(branch).R(:,j);
                d=c.'*x;
                angle=atan2(norm(x-d*c),abs(d));
                constraint(branch)=abs(angle-phi);
            end
            angles=max(constraint);
            scores(k,:)=[err,mtGap,pyGap,correction,angles];
        end
        m=m+1;
        k=numel(sample);
        scores=array2table(scores,'VariableNames',{'branch_value_difference','matlab_gap','python_gap',...
            'signed_target_rounding','matlab_max_angle_error'});
        scores=addvars(scores,repmat(label,k,1),floor((sample(:)-1)/n),mod(sample(:)-1,n),...
            target(sample(:)),gap(sample(:)).','Before',1,...
            'NewVariableNames',{'method','channel_index0','block_index0','target_rad','relative_gap'});
        records{m}=scores;
    end
end
T=vertcat(records{:});
writetable(T,fullfile(out,'frozen_p_matlab_audit.csv'));
assert(max(T.branch_value_difference)<=1e-8 && max(T.matlab_gap)<=1e-8 && ...
    max(T.python_gap)<=1e-8 && max(T.matlab_max_angle_error)<=1e-8);
fprintf('%d frozen P cases, max branch value difference %.8g\n',height(T),max(T.branch_value_difference));
end
