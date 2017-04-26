function  PCs = batchpca( x,k,method,thresh )
% function  PCs = batchpca( x,k )
% x is N*c*1*b
% x has data distributed in the first dimension , second dimensions
% represent vector components.
% OUTPUT:
% batchpca outpus k largest principal components in a matrix k*c*1*b
if nargin<2
    k = 1;
end
%x = x - median(x,1,'omitnan');
xcell = mat2cell(x,size(x,1),size(x,2),size(x,3),ones(1,size(x,4)));
switch method
    case 'maxmed'
        PCs = cellfun(@maxMedpca,xcell,'UniformOutput',false);
        PCs = cat(4,PCs{:});
        PCs = permute(PCs,[2,1,3,4]);
        return;
    case 'purel1'
        PCs = cellfun(@iterpcal1,xcell,'UniformOutput',false);
        PCs = cat(4,PCs{:});
        PCs = permute(PCs,[2,1,3,4]);
        return;
    case 'cor'
        nanflagcell = param2cell('complete',size(xcell));
        nameflagcell = param2cell('rows',size(xcell));
        Covar = cellfun(@mycorr,xcell,nameflagcell,nanflagcell,'UniformOutput',false);
    case 'pca'
        nanflagcell = param2cell('omitrows',size(xcell));
        Covar = cellfun(@cov,xcell,nanflagcell,'UniformOutput',false);
end

Covar = cellfun(@rmnan,Covar,'UniformOutput',false);
[PCs2,~] = cellfun(@eig,Covar,'UniformOutput',false);
PCsMat = cat(4,PCs2{:});
PCs = PCsMat(:,end-k+1:end,:,:);
PCs = permute(PCs,[2,1,3,4]);

% PCs = zeros(1,size(x,2),1,size(x,4),'like',x);
% PCs = sum(PCs,1);
% for i = 1: size(x,4)
% [V,D] = eig( cov( x(:,:,:,i) ));
% PCs(1,:,:,i) = V(:,end)';
% end


end
function C = mycorr(x,varargin)
    x(isnan(x)) = 0;
    C = x'* x;
end
function Cov = rmnan(Cov)
Cov(isnan(Cov)) = 0 ;
end
function Param = param2cell(P,sz)
if numel(sz)<4
    sz = [sz,ones(1,4-numel(sz))];
end
Param = repmat({P},sz(1),sz(2),sz(3),sz(4));
end
