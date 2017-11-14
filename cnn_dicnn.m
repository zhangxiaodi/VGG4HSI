function [net, info] = cnn_dicnn(varargin)
%CNN_DICNN Demonstrates fine-tuning a pre-trained CNN on imagenet dataset
%it is good job in your code!
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;
% 修改读入文件夹的路径
opts.dataDir = fullfile('data','image') ;
opts.expDir  = fullfile('exp', 'image') ;
% 导入预训练的model
opts.modelPath = fullfile('F:\matconvnet-1.0-beta18\models','imagenet-vgg-f.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;

opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct() ;
opts.train.gpus = [];
opts.train.batchSize = 8 ;
opts.train.numSubBatches = 4 ;
opts.train.learningRate = 1e-4 * [ones(1,10), 0.1*ones(1,5)];

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
net = load(opts.modelPath);
% 修改一下这个model
net = prepareDINet(net,opts);
% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
% 准备数据格式
if exist(opts.imdbPath,'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_image_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

imdb.images.set = imdb.images.sets;

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.name ;

% % 求训练集的均值
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage') ;
else
    averageImage = getImageStats(opts, net.meta, imdb) ;
    save(imageStatsPath, 'averageImage') ;
end
% % 用新的均值改变均值
net.meta.normalization.averageImage = averageImage;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
% 索引训练集==1  和测试集==3
opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==3) ;
% 训练
[net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
% 保存训练完的网络
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

net_ = net.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
% bopts.averageImage = []; 
bopts.averageImage = meta.normalization.averageImage ;
% bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;

fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;


% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
% 判断读入数据为训练还是测试
for i = 1:length(batch)
    if imdb.images.set(batch(i)) == 1 %1为训练索引文件夹
        images(i) = strcat([imdb.imageDir.train filesep] , imdb.images.name(batch(i)));
    else
        images(i) = strcat([imdb.imageDir.test filesep] , imdb.images.name(batch(i)));
    end
end;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'input', im, 'label', labels} ;
end

% 求训练样本的均值
% -------------------------------------------------------------------------
function averageImage = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
batch = 1:length(train);
fn = getBatchFn(opts, meta) ;
train = train(1: 100: end);
avg = {};
for i = 1:length(train)-1
    temp = fn(imdb, batch(train(i):train(i)+99)) ;
    temp = temp{2};
    avg{end+1} = mean(temp, 4) ;
end

averageImage = mean(cat(4,avg{:}),4) ;
% 将GPU格式的转化为cpu格式的保存起来（如果有用GPU）
averageImage = gather(averageImage);
