function net = cnn_cifar_init(prenet,varargin) 
opts.networkType = 'simplenn' ; 
opts = vl_argparse(opts, varargin) ; 
isload = ~isempty(prenet);
loadconv = false;
lr = [.1 2] ; 
k =1; 
% Define network CIFAR10-quick 
net.layers = {} ; 
freeze = [false,false,false,false,false];
convfreeze = [false,true];
convDecay = [~convfreeze(1),~convfreeze(2)];
% Block 1 
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 100*eyeconv(5,5,3,k*32, 'single');
    convbias = zeros(1,k*32,'single');
end
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.01*convInit, convbias}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2,...
                           'freeze',convfreeze,...
                           'weightDecay',convDecay) ;
if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
    
else
    freluinit = freluInit(k*32,2, 'single');
end
j = 1;
net.layers{end+1} = struct('type', 'frelu','weights', {{freluinit}},'freeze',freeze(j)...
                          ,'weightDecay',~freeze(j)) ; 
net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; 
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 20*eyeconv(5,5,k*32,k*32, 'single');
    convbias = zeros(1,k*32,'single');
end
 
% Block 2 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*convInit/k, convbias}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2,...
                           'freeze',convfreeze,...
                           'weightDecay',convDecay) ; 
if isload 
    freluinit = loadweight(prenet,numel(net.layers)+1);
   
else
    freluinit = freluInit(k*32,2, 'single');
end 
 j = j+1;
net.layers{end+1} = struct('type', 'frelu','weights', {{freluinit}},'freeze',freeze(j),'weightDecay',~freeze(j)) ; 

net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; % Emulate caffe 

if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 20*eyeconv(5,5,k*32,k*64, 'single');
    convbias = zeros(1,k*64,'single');
end
% Block 3 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*convInit/k, convbias}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2,...
                           'freeze',convfreeze,...
                           'weightDecay',convDecay) ; 
if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
   
else
    freluinit = freluInit(k*64,2, 'single');
end 
 j = j+1;
net.layers{end+1} = struct('type', 'frelu','weights', {{freluinit}},'freeze',freeze(j),'weightDecay',~freeze(j)) ; 
net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; % Emulate caffe 
 
% Block 4 
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 20*eyeconv(4,4,k*64,k*64, 'single');
    convbias = zeros(1,k*64,'single');
end
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*convInit/k,convbias}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 0,...
                           'freeze',convfreeze,...
                           'weightDecay',convDecay) ; 
if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
    
else
    freluinit = freluInit(k*64,2, 'single');
end 
j = j+1;
net.layers{end+1} = struct('type', 'frelu','weights', {{freluinit}},'freeze',freeze(j),'weightDecay',~freeze(j)) ; 

 
% Block 5 
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 20*eyeconv(1,1,k*64,10, 'single');
    convbias = zeros(1,10,'single');
end
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*convInit/k, convbias}}, ... 
                           'learningRate', .1*lr, ... 
                           'stride', 1, ... 
                           'pad', 0,...
                           'freeze',convfreeze,...
                           'weightDecay',convDecay) ; 
 if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
   
else
    freluinit = freluInit(10,2, 'single','halfhalf');
end 
 j = j+1;
net.layers{end+1} = struct('type', 'avr','weights', {{freluinit}},'freeze',freeze(j),'weightDecay',~freeze(j)) ; 
% Loss layer 
net.layers{end+1} = struct('type', 'softmaxloss') ; 
 
% Meta parameters 
net.meta.inputSize = [32 32 3] ; 
net.meta.trainOpts.learningRate = [ 0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)] ; 
net.meta.trainOpts.weightDecay = 0.0001 ; 
net.meta.trainOpts.batchSize = 100 ; 
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ; 
 
% Fill in default values 
net = vl_simplenn_tidy(net) ; 
 
% Switch to DagNN if requested 
switch lower(opts.networkType) 
  case 'simplenn' 
    % done 
  case 'dagnn' 
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ; 
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ... 
             {'prediction','label'}, 'error') ; 
  otherwise 
    assert(false) ; 
end 
end
function [w,w2] = loadweight(net,i)
w=0;
w2=[];
switch net.layers{i}.type
    case 'conv'
        
        w = net.layers{i}.weights{1};
        w2 = net.layers{i}.weights{2};
%         ws = sign(w);
%         w = abs(eyeconv(size(w),'single')).*ws;
    case 'frelu'
        w = net.layers{i}.weights{1};
        w2 = [];

    case 'fbrelu'
end
%w = bsxfun(@rdivide,w,max(abs(w),[],2));
end
function w = freluInit(k,dumb1,dumb2,method)
if nargin<4
method = 'ones';
end
switch method
    case 'rand'
        w= 2*rand(k,2,'single')- 1;
    case 'ones'
        w= ones(k,2,'single');
    case 'halfhalf'
        w = ones(k,2,'single');
        w(:,1) = -w(:,1);
end    
end