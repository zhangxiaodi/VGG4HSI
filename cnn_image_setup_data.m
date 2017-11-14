function imdb = cnn_image_setup_data(varargin)

opts.dataDir = fullfile('data','image') ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% ------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

metaPath = fullfile(opts.dataDir, 'classInd.txt') ;

fprintf('using metadata %s\n', metaPath) ;
tmp = importdata(metaPath);
nCls = numel(tmp);
% �ж�������趨���Ƿ�һ�� 10Ϊ����������������Լ������ݼ���Ҫ�޸ģ�
if nCls ~= 10
  error('Wrong meta file %s',metaPath);
end
% �����ַ������
cats = cell(1,nCls);
for i=1:numel(tmp)
  t = strsplit(tmp{i});
  cats{i} = t{2};
end
% ���ݼ��ļ���ѡ��
imdb.classes.name = cats ;
imdb.imageDir.train = fullfile(opts.dataDir, 'train') ;
imdb.imageDir.test = fullfile(opts.dataDir, 'test') ;

%% -----------------------------------------------------------------
%                                              load image names and labels
% -------------------------------------------------------------------------

name = {};
labels = {} ;
imdb.images.sets = [] ;
%%
fprintf('searching training images ...\n') ;
% ����ѵ������ǩ
train_label_path = fullfile(opts.dataDir, 'train_label.txt') ;
train_label_temp = importdata(train_label_path);
temp_l = train_label_temp.data;
for i=1:numel(temp_l)
    train_label{i} = temp_l(i);
end
if length(train_label) ~= length(dir(fullfile(imdb.imageDir.train, '*.png')))
    error('training data is not equal to its label!!!');
end

i = 1;
for d = dir(fullfile(imdb.imageDir.train, '*.png'))'
    name{end+1} = d.name;
    labels{end+1} = train_label{i} ;
    if mod(numel(name), 10) == 0, fprintf('.') ; end
    if mod(numel(name), 500) == 0, fprintf('\n') ; end
    imdb.images.sets(end+1) = 1;%train
    i = i+1;
end
%%
fprintf('searching testing images ...\n') ;
% �����������ǩ
test_label_path = fullfile(opts.dataDir, 'test_label.txt') ;
test_label_temp = importdata(test_label_path);
temp_l = test_label_temp.data;
for i=1:numel(temp_l)
    test_label{i} = temp_l(i);
end
if length(test_label) ~= length(dir(fullfile(imdb.imageDir.test, '*.png')))
    error('testing data is not equal to its label!!!');
end
i = 1;
for d = dir(fullfile(imdb.imageDir.test, '*.png'))'
    name{end+1} = d.name;
    labels{end+1} = test_label{i} ;
    if mod(numel(name), 10) == 0, fprintf('.') ; end
    if mod(numel(name), 500) == 0, fprintf('\n') ; end
    imdb.images.sets(end+1) = 3;%test
    i = i+1;
end
%%
labels = horzcat(labels{:}) ;
imdb.images.id = 1:numel(name) ;
imdb.images.name = name ;
imdb.images.label = labels ;