function [feat,residuals ] = em_nnmedianfeature( x,maxMed )
%EM_NNMEDIANFEATURE Summary of this function goes here
%   Detailed explanation goes here
method = 'instance';
if nargin> 3
    doder = true;
    feat = [];
else
    doder = false;
    [spacefeat,valfeat,residuals] = extractfeature(x,method,maxMed);
end



end

function [spaceFeat,valFeat,residuals] = extractfeature(x,maxMed)
    valueSet = reshape(x,[],1,1,size(x,4));
    [valFeat,Bins,residuals] =  extractMeds(valueSet,nan,true,maxMed);
    residuals = reshape(residuals,size(x));
    Bins = reshape(Bins,[size(x),size(Bins,5)]);
    % CurrMeds have medians across the first dimension
    [spaceFeat] = compressLocations(Bins,1);
    % Locs is bitnun*M*1*Batch
    % CurrMeds is bitNum*1*1*batch
   % feat = cat(2,CurrMeds,Locs);
end



function LocsMed = compressLocations(Bins,pivot)
% return a bitnum*M*1*Batch tensor
% pivot sets which binary values to look for. [0,1,nan]
    if ~isnan(pivot)
        pivotMap = Bins == pivot;
    else
        pivotMap = isnan(Bins);
    end
    Bins(~pivotMap) = nan;
    Bins(pivotMap) = 1;


    curBin = Bins;
    IndsRef = (1:size(Bins,1))';
    Inds = curBin.*IndsRef;
    [ IndsMeds,~,~] = extractMeds(Inds,nan,false,[]);
    % IndsMeds for bit i , has medians of locations stored in the dim 1
    LocsMed = permute(IndsMeds,[5,1,3,4,2]);
    % IndsMeds for bit i , has medians of locations stored in the dim 2
    %LocsMed = catBank(LocsMed,IndsMeds,2,1);
end







function Bank = catBank(Bank,Instance,vecdim,instdim)
% attaches Meds to bank in along dimension dims(2) where dims(1) is the
% vector
if numel(Bank) == 0
    Bank = Instance;
    return;
end
ins_vec_sz = size(Instance,vecdim);
bank_vec_sz = size(Bank,vecdim);
pad_sz = abs(ins_vec_sz - bank_vec_sz);
is_pad_bank = bank_vec_sz<ins_vec_sz;
% turns first dim as vec dim and second dim as instance dim
Bank = permute(Bank,[vecdim,instdim,3,4]);
Instance = permute(Instance,[vecdim,instdim,3,4]);
    switch class(Bank)
            case 'gpuArray'
                if is_pad_bank
                    PAD = gpuArray.zeros(pad_sz,size(Bank,2),size(Bank,3),size(Bank,4))*nan;
                else
                    PAD = gpuArray.zeros(pad_sz,size(Instance,2),size(Instance,3),size(Instance,4))*nan;
                end
        otherwise
                if is_pad_bank
                    PAD = zeros(pad_sz,size(Bank,2),size(Bank,3),size(Bank,4))*nan;
                else
                    PAD = zeros(pad_sz,size(Instance,2),size(Instance,3),size(Instance,4))*nan;
                end
    end
if is_pad_bank
    Bank = cat(1,Bank,PAD);
else
    Instance = cat(1,Instance,PAD);
end
Bank = cat(2,Bank,Instance);
Bank = ipermute(Bank,[vecdim,instdim,3,4]);

%%
% B1 = size(Bank,vecdim);
% if B1 == 0
%     Bank = Instance;
%     return;
% end
% M1 = size(Instance,vecdim);
% if size(Instance,1) > size(Bank,1)
%     padNum = M1-B1;
%     switch class(Bank)
%         case 'gpuArray'
%             PAD = gpuArray.zeros(padNum,size(Bank,2),size(Bank,3),size(Bank,4));
%         otherwise
%             PAD = zeros(padNum,size(Bank,2),size(Bank,3),size(Bank,4));
%     end
%     Bank = cat(vecdim,Bank,PAD);
%     Bank = cat(instdim,Bank,Instance);
% else
%     Bank = cat(instdim,Bank,Bank(:,1,:,:));
%     Bank(1:size(Instance,1),end,:,:) = Instance;
% end
    
end

function df  = flatten(data,D)
% flatten puts the variables in the dimensions specified by D in the first
% dimension
% D is an integer 
end

function y = derv(x)
w = gpuArray(single([1;-1]));
y = vl_nnconv(x,w,[],'stride',2);
end

function Int = GreytoReal(x)
Sz = size(x);
bitNum = size(x,5);
switch class(x)
    case 'gpuArray'
        Int = gpuArray.zeros(Sz(1:4));
    otherwise
        Int = zeros(Sz(1:4));
end
for i = 1: bitNum
   Int = Int +  x(:,:,:,:,i)* 2^(-i); 
end
end