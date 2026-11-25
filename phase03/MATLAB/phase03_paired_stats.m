function phase03_paired_stats(runDir)
% 凍結 P 目標的全區塊等角度比較，保存 G 可行子集及完整分母。
arguments
    runDir (1,1) string
end
source=fullfile(runDir,'04_影像與矩陣','matched_targets.h5');
out=fullfile(runDir,'05_統計分析');
names=["S","G","R","P"];
loss=cell(8,2,4);
relative=cell(8,2);
i=0;
for side=["U","V"]
    for j=1:4
        i=i+1;
        vector=side+j;
        X=reshape(h5read(source,"/P"+vector+"/loss"),4,[]).';
        bound=reshape(h5read(source,"/P"+vector+"/certificate_loss_tolerance"),[],1);
        G=isfinite(X(:,2));
        for s=1:2
            for k=1:4
                keep=isfinite(X(:,k));
                if s==2, keep=keep&G; end
                summary=describe(X(keep,k));
                scope="all_P_targets";
                if s==2, scope="G_feasible"; end
                loss{i,s,k}=addvars(summary,vector,names(k),scope,'Before',1,...
                    'NewVariableNames',{'vector','method','scope'});
            end
        end
        for k=2:3
            positive=X(:,k)>1e-12;
            eligible=positive & X(:,k)>100*bound;
            d=100*(1-X(eligible,4)./X(eligible,k));
            summary=describe(d);
            raw=100*(1-X(positive,4)./X(positive,k));
            summary.count_above_absolute_floor=numel(raw);
            summary.mean_without_certificate_filter=mean(raw);
            summary.min_without_certificate_filter=NaN;
            if ~isempty(raw), summary.min_without_certificate_filter=min(raw); end
            relative{i,k-1}=addvars(summary,vector,names(k),'Before',1,...
                'NewVariableNames',{'vector','baseline'});
        end
    end
end
writetable(vertcat(loss{:}),fullfile(out,'same_target_loss_matlab.csv'));
writetable(vertcat(relative{:}),fullfile(out,'same_target_reduction_matlab.csv'));
end

function row=describe(x)
count=numel(x);
meanValue=mean(x);
q=nan(3,1);
if count>0
    x=sort(x);
    h=1+(count-1)*[.10;.50;.90];
    a=floor(h); b=ceil(h);
    q=x(a)+(h-a).*(x(b)-x(a));
end
row=table(count,meanValue,q(1),q(2),q(3),'VariableNames',{'count','mean','p10','median','p90'});
end
