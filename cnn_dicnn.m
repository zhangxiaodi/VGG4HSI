function [net, info] = cnn_dicnn(varargin)
%CNN_DICNN Demonstrates fine-tuning a pre-trained CNN on imagenet dataset
%it is good job in your code!
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;
% �޸Ķ����ļ��е�·��
opts.dataDir = fullfile('data','image') ;
opts.expDir  = fullfile('exp', 'image') ;
% ����Ԥѵ����model
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
% �޸�һ�����model
net = prepareDINet(net,opts);
% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
% ׼�����ݸ�ʽ
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

% % ��ѵ�����ľ�ֵ
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage') ;
else
    averageImage = getImageStats(opts, net.meta, imdb) ;
    save(imageStatsPath, 'averageImage') ;
end
% % ���µľ�ֵ�ı��ֵ
net.meta.normalization.averageImage = averageImage;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
% ����ѵ����==1  �Ͳ��Լ�==3
opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==3) ;
% ѵ��
[net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
% ����ѵ���������
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
% �ж϶�������Ϊѵ�����ǲ���
for i = 1:length(batch)
    if imdb.images.set(batch(i)) == 1 %1Ϊѵ�������ļ���
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

% ��ѵ�������ľ�ֵ
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
% ��GPU��ʽ��ת��Ϊcpu��ʽ�ı����������������GPU��
averageImage = gather(averageImage);
