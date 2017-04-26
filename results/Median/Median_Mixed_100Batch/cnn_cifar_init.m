function net = cnn_cifar_init(prenet,varargin)
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;
isload = ~isempty(prenet);
loadconv = false;
lr = [.1 2] ;
k =1;
chk =2;
% Define network CIFAR10-quick
medianMethod = 'superMedian';
net.layers = {} ;
freeze = ~[false,false,false,false,false,false,false];
freluDecay = false;
convfreeze = [false,false];
convDecay = [~convfreeze(1),~convfreeze(2)];
featureSize =1000;
%% Block 1 ----------------------------------------------------------------------------------------------------------------------------
net.layers{end+1} = struct('type', 'medianfeat','method','superMedian','weights',{{0.05*randn(1,1,featureSize*3,10, 'single')}}, 'vectorSize',featureSize,'set','spatial');

if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 0.05*randn(16,32*32*3 -1,1,10, 'single');
    convbias = zeros(1,10,'single');
end
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*convInit/k, convbias}}, ...
    'learningRate', .1*lr, ...
    'stride', 1, ...
    'pad', 0,...
    'freeze',convfreeze,...
    'weightDecay',convDecay) ;

 net.layers{end+1} = struct('type', 'relu')
%% Block 5

if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 0.05*randn(1,1,10,10, 'single');
    convbias = zeros(1,10,'single');
end
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*convInit/k, convbias}}, ...
    'learningRate', .1*lr, ...
    'stride', 1, ...
    'pad', 0,...
    'freeze',convfreeze,...
    'weightDecay',convDecay) ;
% Activation 5 ******************************
method = 'cart';
activation = [];
activLR = 4;
if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
    
else
    freluinit = freluInit(10,2, 'single',[method],activation);
end

if ~isempty(activation)
    net.layers{end+1} = struct('type', activation,...
        'weights', {{freluinit}}...
        ,'freeze',freeze(j)...
        ,'weightDecay',freluDecay...
        ,'learningRate', activLR ...
        ,'method',method) ;
end


% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [ 1*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
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
function w = freluInit(k,dumb1,dumb2,method,activation)
if isempty(activation)
    warning('activation not assigned');
    w= [];
    return;
end
if nargin<4
    method = 'polar';
end
switch method
    case 'polar2'
        w= rand(k,1)*2*pi;
    case 'polar1'
        w= rand(k,1)*2*pi*0;
    case 'polar1f'
        w= (rand(k,1)*pi/2);
        Rand = rand(size(w));
        pole = 2*(Rand>0.5) -1;
        w = w + pole*(pi/2);
    case 'cart'
        switch activation
            case 'fbrelu'
                w = compileWfBrelu((2*rand(k,2,'single')- 1)/2,'cartrot');
                %w= (2*rand(k,4,'single')- 1)/2;
            case 'frelu'
                w= (2*rand(k,2,'single')- 1)/2;
            otherwise
                warning('Invalid Activation')
                w= (2*rand(k,4,'single')- 1)/2;
        end
    case 'cartrot'
        % cartesian and rotated
        w= (2*rand(k,2,'single')- 1);
    case 'cartconst'
        w= (2*rand(k,2,'single')- 1);
    case 'ones'
        w= ones(k,1,'single');
        
    case 'halfhalf'
        w= ones(k,2,'single')/2;
    case 'cartl1rot'
        w = rand(k,2);
        w(:,2) = w(:,2)*2*pi;
        
end
end