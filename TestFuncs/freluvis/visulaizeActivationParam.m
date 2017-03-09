function [] = visulaizeActivationParam(net,parNum)
if nargin<2; parNum = [];end
lNum = numel(net.layers);
j = 1;

for i = 1: lNum
    L = net.layers{i};
    prevconvweights = net.layers{i-1}.weights;
    switch L.type
        case 'frelu'
            w = L.weights{1};
            if ~isempty(parNum)
                parNum = min(parNum,size(w,1));
                w = w(1:parNum,:);
            end
            figure(3);
            w= compileWfrelu(w,L.method);
            plotfrelu(w,j,L.type)
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
            plotrelugraph(w,prevconvweights,j,blocks,type);
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
     %   scatter(w(:,3),w(:,4),18,c,'filled');
      %  scatter(w(Ind,3),w(Ind,4),36,'r','+');
   %     scatter(w(Ind,1),w(Ind,2),36,'r','+');
        axis([-1.5,1.5,-1.5,1.5])
        %axis equal
        hold off;
        
end
end
function [] = plotrelugraph(w,prevconvweights,freluId,blocks)
subplot(3,2,freluId);
convs = prevconvweights{1};
bias = prevconvweights{2};
Mat = convMat2mat(prevconvweights,blocks);
Sim = calcSimilarity(Mat,'cosine');
if size(w,1) ~= size(Mat,1)
    % bfrelu
else
    %frelu
end
plot
end
function out = calcSimilarity(mat,method)
switch method
    case 'cosine'
        out = mat * mat';
        norm = sqrt(sum(mat.^2,2));
        out = bsxfun(@rdivide,out,norm);
        out = bsxfun(@rdivide,out,norm');
    case 'dot'
        out = mat* mat'; 
end
end
