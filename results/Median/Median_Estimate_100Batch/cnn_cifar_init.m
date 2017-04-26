function net = cnn_cifar_init(prenet,varargin)
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;
isload = ~isempty(prenet);
loadconv = false;
lr = [.1 2] ;
k =1;
chk =2;
% Define network CIFAR10-quick
medianMethod = 'estimate';
net.layers = {} ;
freeze = ~[false,false,false,false,false,false,false];
freluDecay = false;
convfreeze = [false,false];
convDecay = [~convfreeze(1),~convfreeze(2)];
%% Block 1 ----------------------------------------------------------------------------------------------------------------------------
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 100*eyeconv(5,5,3,k*32, 'single')/k;
    convbias = zeros(1,k*32,'single');
end
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.01*convInit, convbias}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 2, ...
    'freeze',convfreeze,...
    'weightDecay',convDecay) ;
% Activation 1 ******************************
method = 'polar1';
activation = 'birelu';
activLR = 4;
if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
    
else
    freluinit = freluInit(k*32,2, 'single',method,activation);
end
j = 1;
net.layers{end+1} = struct('type', 'median','method',medianMethod,'weights', {{zeros(1,1,k*32,'single')}});
net.layers{end+1} = struct('type', activation,...
    'weights',{{freluinit}},...
    'freeze',freeze(j) ,...
    'weightDecay',freluDecay,...
    'learningRate', activLR, ...
    'method',method) ;

net.layers{end+1} = struct('type', 'pool', ...
    'method', 'avg', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]) ;

%% Block 2 ----------------------------------------------------------------------------------------------------------------------------
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 20*eyeconv(5,5,chk*k*32,k*32, 'single')/chk;
    convbias = zeros(1,k*32,'single');
end


net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*convInit/k, convbias}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 2,...
    'freeze',convfreeze,...
    'weightDecay',convDecay) ;
% Activation 2 ******************************
method = 'polar1';
activation = 'birelu';
activLR =4;
if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
    
else
    freluinit = freluInit(k*32,2, 'single',method,activation);
end
j = j+1;
net.layers{end+1} = struct('type', 'median','method',medianMethod,'weights', {{zeros(1,1,k*32,'single')}});
net.layers{end+1} = struct('type', activation,...
    'weights', {{freluinit}},...
    'freeze',freeze(j),...
    'weightDecay',freluDecay,...
    'learningRate', activLR, ...
    'method',method) ;



net.layers{end+1} = struct('type', 'pool', ...
    'method', 'avg', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]) ; % Emulate caffe
%% Block 3 ----------------------------------------------------------------------------------------------------------------------------
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 20*eyeconv(5,5,chk*k*32,k*64, 'single')/chk;
    convbias = zeros(1,k*64,'single');
end

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*convInit/k, convbias}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 2,...
    'freeze',convfreeze,...
    'weightDecay',convDecay) ;

% Activation 3 ******************************
method = 'polar1';
activation = 'birelu';
activLR = 4;
if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
    
else
    freluinit = freluInit(k*64,2, 'single',method,activation);
end
j = j+1;
net.layers{end+1} = struct('type', 'median','method',medianMethod,'weights', {{zeros(1,1,k*64,'single')}});
net.layers{end+1} = struct('type', activation,...
    'weights', {{freluinit}}...
    ,'freeze',freeze(j)...
    ,'weightDecay',freluDecay...
    ,'learningRate', activLR ...
    ,'method',method) ;


net.layers{end+1} = struct('type', 'pool', ...
    'method', 'avg', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]) ; % Emulate caffe

%% Block 4
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 20*eyeconv(4,4,chk*k*64,k*64, 'single')/chk;
    convbias = zeros(1,k*64,'single');
end
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{0.05*convInit/k,convbias}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 0,...
    'freeze',convfreeze,...
    'weightDecay',convDecay) ;
net.layers{end+1} = struct('type', 'median','method',medianMethod,'weights', {{zeros(1,1,k*64,'single')}});
% Activation 4 ******************************
method = 'polar1';
activation= 'birelu';
activLR = 4;
chk = 2;
if isload
    freluinit = loadweight(prenet,numel(net.layers)+1);
    
else
    freluinit = freluInit(k*64,2, 'single',method,activation);
end
j = j+1;
net.layers{end+1} = struct('type', activation...
    ,'weights', {{freluinit}}...
    ,'freeze',freeze(j),...
    'weightDecay',freluDecay...
    ,'learningRate', activLR ...
    ,'method',method) ;


%% Block 5
if isload && loadconv
    [convInit,convbias] = loadweight(prenet,numel(net.layers)+1);
else
    convInit = 20*eyeconv(1,1,chk*k*64,10, 'single')/chk;
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
j = j+1;

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