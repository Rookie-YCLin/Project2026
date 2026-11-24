function main_phase01(runDir, isPilot)
% 從 Python 產生的矩陣獨立計算 SVD、擾動、統計及相關。
arguments
    runDir (1,1) string
    isPilot (1,1) logical = false
end
inputFile = fullfile(runDir, '02_原始矩陣', 'block_matrices.h5');
svdFile = fullfile(runDir, '03_SVD分解結果', 'svd_matlab.h5');
metricFile = fullfile(runDir, '04_逐區塊擾動數據', 'perturbation_matlab.h5');
statsDir = fullfile(runDir, '05_統計分析');
logDir = fullfile(runDir, '10_執行紀錄');
assert(~isfile(svdFile) && ~isfile(metricFile), '請使用新執行資料夾。');
codeRoot = fileparts(fileparts(mfilename('fullpath')));
cfg = jsondecode(fileread(fullfile(codeRoot, 'Configs', 'phase01.json')));
quality = 1:100;
if isPilot
    quality = cfg.pilot_qf(:).';
end
qgroups = compose("QF%03d", quality);
groups = ["/identity", "/jpeg/" + qgroups];
labels = ["identity", qgroups];
synth = validate_numerical_results();
A = loadPages(inputFile, '/original/A');
n = size(A,3)/3;
grid = double(h5read(inputFile,'/index/grid'));
assert(n == prod(grid));
preprocess = checkPreprocess(inputFile, '/original', A, n);
F = factor(A);
writeFactor(svdFile, '/original', F, n);
sV = strings(numel(groups), 1);
test = zeros(numel(groups), 4);
st = cell(size(groups)); co = st; bi = st; dg = st; sens = st;
near = cfg.relative_near_threshold;
for k = 1:numel(groups)
    group = groups(k);
    B = loadPages(inputFile, group + "/A");
    G = factor(B);
    if k == 1
        assert(isequal(A,B));
    else
        writeFactor(svdFile, group, G, n);
        if ismember(quality(k-1),cfg.pilot_qf)
            preprocess = max(preprocess, checkPreprocess(inputFile, group, B, n));
        end
    end
    [M,D,checks] = compute_perturbation(A,B,F,G,near);
    if k == 1
        assert(max(M.D_F) == 0 && max(M.D_sigma) == 0);
        limit = acosd(max(0,1-max(F.orthogonality,G.orthogonality)-8*eps));
        fprintf('Identity max angle: %.9g degree (bound %.9g)\n',max([M.theta_u M.theta_v],[],'all'),limit);
        assert(max([M.theta_u M.theta_v],[],'all') <= limit);
    end
    [metricNames, metricValues] = unpack(M);
    [flagNames, flagValues] = unpack(D);
    saveTableData(metricFile, group + "/metrics", metricNames, metricValues, n);
    saveTableData(metricFile, group + "/diagnostics", flagNames, uint8(flagValues), n);
    [st{k},co{k},bi{k},dg{k},sens{k}] = compute_statistics(labels(k),M,metricNames,metricValues,flagNames,flagValues,n,near);
    sV(k) = labels(k);
    test(k,:) = [G.reconstruction, G.orthogonality, checks];
    fprintf('MATLAB independently analyzed %s, %d blocks\n',labels(k),3*n);
end
writetable(vertcat(st{:}), fullfile(statsDir,'statistics_matlab.csv'));
writetable(vertcat(co{:}), fullfile(statsDir,'correlations_matlab.csv'));
writetable(vertcat(bi{:}), fullfile(statsDir,'gap_bins_matlab.csv'));
writetable(vertcat(dg{:}), fullfile(statsDir,'diagnostics_matlab.csv'));
writetable(vertcat(sens{:}), fullfile(statsDir,'identifiable_angles_matlab.csv'));
T = array2table(test,'VariableNames',{'svd_residual','orthogonality','frobenius_delta','mirsky_violation'});
T.condition = sV;
writetable(T,fullfile(logDir,'numerical_checks_matlab.csv'));
meta = struct('version',version,'release',version('-release'),'products',ver, ...
    'timestamp_utc',char(datetime('now','TimeZone','UTC','Format','uuuu-MM-dd HH:mm:ss Z')), ...
    'computer',computer,'n_per_channel',n,'grid',grid,'available_plot_font', ...
    any(strcmpi(listfonts,'Times New Roman')), ...
    'original_svd_residual',F.reconstruction,'original_orthogonality',F.orthogonality, ...
    'preprocess_relative_max',preprocess,'synth',synth);
fid = fopen(fullfile(logDir,'environment_matlab.json'),'w','n','UTF-8');
assert(fid~=-1);
fprintf(fid,'%s\n',jsonencode(meta,'PrettyPrint',true));
fclose(fid);
end

function pages = loadPages(file,path)
% HDF5 的 C 維度被 MATLAB 逆向讀取，再換回矩陣的 row,column。
raw = h5read(file,path);
assert(size(raw,1)==4 && size(raw,2)==4 && size(raw,4)==3);
pages = reshape(permute(raw,[2 1 3 4]),4,4,[]);
end

function err = checkPreprocess(file, group, pages, n)
sentinel = h5read(file,'/layout_sentinel');
assert(isequal(size(sentinel),[4 4 2 3]));
for c = 0:2
    for i = 0:1
        for r = 0:3
            for k = 0:3
                assert(sentinel(k+1,r+1,i+1,c+1) == 1000*c+100*i+10*r+k);
            end
        end
    end
end
p = group + "/preprocess_check/";
indices = double(h5read(file,p+"block"))+1;
shuffler = double(h5read(file,'/index/shuffle'))+1;
bgr = double(permute(h5read(file,p+"bgr"),[2 1 3 4]));
yuvReference = double(permute(h5read(file,p+"yuv"),[2 1 3 4]));
llReference = double(permute(h5read(file,p+"LL"),[2 1 3 4]));
D = cos(pi*((0:3).')*((0:3)+0.5)/4)*sqrt(2/4);
D(1,:) = 1/2;
err = zeros(1,4);
for k = 1:numel(indices)
    pixels = bgr(:,:,:,k);
    blue = pixels(:,:,1); green = pixels(:,:,2); red = pixels(:,:,3);
    y = 0.299*red + 0.587*green + 0.114*blue;
    yuv = cat(3,y,0.492*(blue-y)+0.5,0.877*(red-y)+0.5);
    ref = yuvReference(:,:,:,k);
    err(1) = max(err(1),max(abs(yuv-ref),[],'all')/max(1,max(abs(ref),[],'all')));
    ll = (yuv(1:2:end,1:2:end,:)+yuv(1:2:end,2:2:end,:)+ ...
          yuv(2:2:end,1:2:end,:)+yuv(2:2:end,2:2:end,:))/2;
    ref = llReference(:,:,:,k);
    err(2) = max(err(2),max(abs(ll-ref),[],'all')/max(1,max(abs(ref),[],'all')));
    for c = 1:3
        B = D*ll(:,:,c)*D.';
        C = D*ref(:,:,c)*D.';
        b = reshape(B.',[],1);
        a = reshape(C.',[],1);
        sequence = shuffler(:,indices(k));
        refA = pages(:,:,indices(k)+(c-1)*n);
        scale = max(1,norm(refA,'fro'));
        err(3) = max(err(3),norm(reshape(a(sequence),4,4).'-refA,'fro')/scale);
        err(4) = max(err(4),norm(reshape(b(sequence),4,4).'-refA,'fro')/scale);
    end
end
% 原版單精度色彩轉換、DWT、DCT 的步驟誤差界，以 32 個 single epsilon 核對。
assert(all(err <= 32*double(eps('single'))));
end

function writeFactor(file, path, F, n)
writeData(file,path+"/U",permute(reshape(F.U,4,4,n,3),[2 1 3 4]),'channel,block,row,column');
writeData(file,path+"/sigma",reshape(F.s.',4,n,3),'channel,block,singular_index');
writeData(file,path+"/V",permute(reshape(F.V,4,4,n,3),[2 1 3 4]),'channel,block,row,column');
end

function saveTableData(file, path, names, values, n)
writeData(file,path,reshape(values.',numel(names),n,3),'channel,block,field');
h5writeatt(file,path,'fields',char(strjoin(names,",")));
end

function writeData(file,path,data,axes)
shape = size(data);
chunk = shape;
chunk(end-1) = min(shape(end-1),1024);
chunk(end) = 1;
h5create(file,path,shape,'Datatype',class(data),'ChunkSize',chunk,'Deflate',1,'Shuffle',true);
h5write(file,path,data);
h5writeatt(file,path,'axes',axes);
h5writeatt(file,path,'dtype',class(data));
h5writeatt(file,path,'indices','HDF5 0-based; Y,U,V; block row-major; SVD j=1..4 in math');
end

function [names,values] = unpack(data)
keys = fieldnames(data);
names = strings(0,1);
values = [];
for i = 1:numel(keys)
    key = string(keys{i});
    v = data.(key);
    if size(v,2) > 1
        names = [names; (key + "_" + (1:size(v,2))).']; %#ok<AGROW>
    else
        names = [names; key]; %#ok<AGROW>
    end
    values = [values, v]; %#ok<AGROW>
end
end
