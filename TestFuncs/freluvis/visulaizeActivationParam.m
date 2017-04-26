function [] = visulaizeActivationParam(net,parNum)
if nargin<2; parNum = [];end
lNum = numel(net.layers);
j = 1;

for i = 1: lNum
    L = net.layers{i};
    
    if i~=1
    prevconvweights = net.layers{i-1}.weights;
    if ~isfield(net.layers{i-1},'block')
        blocks = 1;
    else
        blocks = net.layers{i-1}.block;
    end
    end
    switch L.type
        case 'frelu'
            w = L.weights{1};
            if ~isempty(parNum)
                parNum = min(parNum,size(w,1));
                w = w(1:parNum,:);
                Conv = prevconvweights{1};
                Bias = prevconvweights{2};
                Conv = Conv(:,:,:,1:parNum);
                Bias = Bias(1:parNum);
                prevconvweights{1} = Conv;
                prevconvweights{2} = Bias;
            end
            figure(3);
            w= compileWfrelu(w,L.method);
            plotfrelu(w,j,L.type);
            plotrelugraph(w,prevconvweights,j,1);
            j = j+1;
        case 'fbrelu'
            w= L.weights{1};
           
            if ~isempty(parNum)
                parNum = min(parNum,size(w,1));
                w = w(1:parNum,:);
            end
            w= compileWfBrelu(w,L.method);
           % w = w(:,1:2);
           
            figure(3);
            plotfrelu(w,j,L.type);
            Convs = prevconvweights{1};
            Bias = prevconvweights{2};
            Convs = cat(4,Convs,Convs);
            Bias = cat(2,Bias,Bias);
            prevconvweights{1} = Convs;
            prevconvweights{2} = Bias;
            w = cat(1,w(:,1:2),w(:,3:4));
            frame = plotrelugraph(w,prevconvweights,j,blocks);
            j = j+1;
    end
    
end
end
function [] = plotfrelu(w,lnum,type)
w = gather(w);
diffbrelu = true;
ax =subplot(3,2,lnum);
switch type
    case 'frelu'
        c = linspace(1,100,size(w,1));
        scatter(w(:,1),w(:,2),[],c,'filled');title([type,int2str(lnum)]);
       
        ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
        axis([-1.5,1.5,-1.5,1.5])
        %axis equal
    case 'fbrelu'
        
        
        c = linspace(1,100,size(w,1));
        Ind = randperm(size(w,1),1);
        if ~diffbrelu
        scatter(w(:,1),w(:,2),18,c,'filled');hold on;
        else
            scatter(w(:,1)-w(:,3),w(:,2)-w(:,4),18,c,'filled');hold on;

        end
        ax = gca;
        ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
     %   scatter(w(:,3),w(:,4),18,c,'filled');
      %  scatter(w(Ind,3),w(Ind,4),36,'r','+');
   %     scatter(w(Ind,1),w(Ind,2),36,'r','+');
        %axis([-1.5,1.5,-1.5,1.5])
       
        hold off;
        
end
end
function frame = plotrelugraph(w,prevconvweights,freluId,blocks)
MaxLineWidth = 2;
convs = prevconvweights{1};
bias = prevconvweights{2}';
Mat = convMat2mat(convs,blocks);
Sim = calcSimilarity(Mat,'cosine');
scrsz = get(groot,'ScreenSize');
map = createColorMap([0,0,1],[1,0,0],128);
figure(freluId+4);title(['frelu',int2str(freluId)]);

if size(w,1) ~= size(Mat,1)
    % bfrelu
else
    %frelu
    %set(gcf, 'Position', get(0, 'Screensize'));
    thresh =0.2;
    Sim(abs(Sim)<thresh) = 0;
    Sim = gather(Sim);
    G = graph(Sim,'OmitSelfLoops');
    positions = gather(cat(2,w,bias));
    Weights = G.Edges.Weight;
    color = Weights>0;
    color = (color *63) +1;
    LineWidths = abs(Weights)*MaxLineWidth;
    colormap(map)
    caxis([-1,1]);
    p = plot(G,'XData',positions(:,1),'YData',positions(:,2),'ZData',positions(:,3),...
        'LineWidth',abs(LineWidths),'EdgeCData',Weights);
    caxis([-1,1]);
    
    ax = gca;
    colormap(ax,map);
    view(2);
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    drawnow;
    frame = getframe(ax);
    
end
end

function out = calcSimilarity(mat,method)
switch method
    case 'cosine'
        norm = sqrt(sum(mat.^2,2));
        mat = bsxfun(@rdivide,mat,norm);
        out = mat * mat';
        
       % out = bsxfun(@rdivide,out,norm);
        %out = bsxfun(@rdivide,out,norm');
    case 'dot'
        out = mat* mat'; 
end
end
function rgbmap = createColorMap(negative,positive,Num)
hsvmap = zeros(Num*2,3);
[hueNeg,~,~] = rgb2hsv(negative);
[huePos,~,~] = rgb2hsv(positive);
hsvmap(:,3) = 1;
hsvmap(1:end/2,1) = hueNeg;
hsvmap(end/2+1 : end,1) = huePos;
hsvmap(1:end/2,2) = linspace(1,0,Num);
hsvmap(end/2 + 1 :end,2) = linspace(0,1,Num);


rgbmap = hsv2rgb(hsvmap);
end