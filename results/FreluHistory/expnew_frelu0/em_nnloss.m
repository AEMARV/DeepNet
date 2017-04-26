function [y,percentErr] = em_nnloss(l,x,c,prevpercent,dzdy)
%VL_NNSOFTMAXLOSS CNN combined softmax and logistic loss.
%   **Deprecated: use `vl_nnloss` instead**
%
%   Y = VL_NNSOFTMAX(X, C) applies the softmax operator followed by
%   the logistic loss the data X. X has dimension H x W x D x N,
%   packing N arrays of W x H D-dimensional vectors.
%
%   C contains the class labels, which should be integers in the range
%   1 to D. C can be an array with either N elements or with dimensions
%   H x W x 1 x N dimensions. In the fist case, a given class label is
%   applied at all spatial locations; in the second case, different
%   class labels can be specified for different locations.
%
%   DZDX = VL_NNSOFTMAXLOSS(X, C, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% work around a bug in MATLAB, where native cast() would slow
% progressively
revtype = 'perclass';
doder = nargin>4;
if doder
    doder=doder;
end
if isa(x, 'gpuArray')
  switch classUnderlying(x) ;
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
else
  switch class(x)
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
end

%X = X + 1e-6 ;
sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
if numel(c) == sz(4)
  % one label per image
  c = reshape(c, [1 1 1 sz(4)]) ;
end
if size(c,1) == 1 & size(c,2) == 1
  c = repmat(c, [sz(1) sz(2)]) ;
end

percentErr = calcErrorPercent(x,c);

switch revtype
    case 'perclass'
        
    case 'allclass'
        percentErr = calcErrorPercent(x,c);
        Rate = calcRate(percentErr,prevpercent);
    case 'distlabel'
        if doder
            Rate = calcRate(percentErr,prevpercent);
            [c,I] = distLabel(x,c,Rate);
        end
    otherwise
        error('unknown');
end
% one label per spatial location
sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(sz_, [sz(1) sz(2) sz_(3) sz(4)])) ;
assert(sz_(3)==1 | sz_(3)==2) ;

% class c = 0 skips a spatial location
mass = cast(c(:,:,1,:) > 0) ;
if sz_(3) == 2
  % the second channel of c (if present) is used as weights
  mass = mass .* c(:,:,2,:) ;
  c(:,:,2,:) = [] ;
end

% convert to indexes
truec = c;
c = c - 1 ;
c_ = 0:numel(c)-1 ;
c_ = 1 + ...
  mod(c_, sz(1)*sz(2)) + ...
  (sz(1)*sz(2)) * max(c(:), 0)' + ...
  (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;

% compute softmaxloss

[xmax,calcc] = max(x,[],3) ;
ex = exp(bsxfun(@minus, x, xmax)) ;

%n = sz(1)*sz(2) ;
if ~doder
  t = xmax + log(sum(ex,3)) - reshape(x(c_), [sz(1:2) 1 sz(4)]) ;
  y = sum(sum(sum(mass .* t,1),2),4) ;
else
  loss= xmax + log(sum(ex,3)) - reshape(x(c_), [sz(1:2) 1 sz(4)]) ;
  y = bsxfun(@rdivide, ex, sum(ex,3)) ;
  y(c_) = y(c_) - 1;
  y = bsxfun(@times, y, bsxfun(@times, mass, dzdy)) ;
  switch revtype
      case 'perclass'
          optype = 'negate';
          RateType = 'double';
          choiceType ='random';
          p = 0.5;
          [errNumPerClass,~,totals,~,correctsMat] = calcErrorPerClass(calcc,truec);
          negNum = calcNegNumPerClass(errNumPerClass,totals,RateType);
          Inds = calcNegInds(loss,correctsMat,negNum,choiceType);
          switch optype
              case 'negate'
                  if rand<p
                    y(:,:,:,Inds) = -y(:,:,:,Inds);  
                  end
              otherwise
             
          end
      case 'distlabel'
          y(:,:,:,I) = -y(:,:,:,I);
      case 'allclass'
          y = randomNegate(y,Rate,c+1,x,loss);
      otherwise
          error('unknnown');
  end

end
end




function Inds = revInds(truec,calcc,loss,method)
%% function Inds = revInds(truec,calcc,loss,method)
% truec is the true class labels, 1,1,1,C
% calcc is the calculated class lables , 1,1,1,C
% loss is the calculated Loss
% method can be less,more,random where less negates the least loss and more
% is the opposite. random sets the first 
end

%% allclass funcs
function y=  calcErrorPercent(x,c)
 [M,I] =   max(x,[],3);
 errs = ((c)~=I);
 y = sum(errs,4)/numel(c);
 
end
function p = calcRate(percentErr,prevpercent)
 if ~isempty(prevpercent)
      diff = percentErr - prevpercent;
      diff = vl_nnrelu(diff);
      if diff ~= 0
          diff = diff/(1-percentErr);
      end
 else
      prevpercent = 0;
      diff = 0;
 end
 

 if rand(1)>0.5
 % p = (percentErr^(1)/((1-percentErr)^(1)));
% p = percentErr;
p = 0 ;
 else
   p = percentErr;
 end
 
% p = percentErr;
end
function y = randomNegate(y,p,c,x,Loss)
Random = false;
Sort = true;


negNum = ceil(size(x,4)*p);

[M,I] =   max(x,[],3);
 corrects = ((c)==I );
 errs = c~=I;
 if Random
Rands = gpuArray.rand(1,1,1,numel(find(corrects)));
Pos = Rands>=p;
 else
     negNum = floor(numel(find(corrects))*min(p,1));
     Rands = gpuArray.ones(1,1,1,numel(find(corrects)));
     
     Rands(1:negNum) = 0;
     if Sort
     
     [~,i] = sort(-Loss(corrects));
     [~,ip] = sort(i);
     Rands = Rands(1,1,1,ip);
     end
     Pos = Rands;
 end
Sign = (2*Pos -1);
y(:,:,:,corrects) = bsxfun(@times,Sign,y(:,:,:,corrects));
 %y = x;

end


%% perclass funcs
function [numErrsPerClass,errors,totalsPerClass,corrects,correctsMat] = calcErrorPerClass(cCalced,trueC)
%% numErrsPerClass is 1*C and has the number of errors per class
% errors is n*1 with the errors being the true class number and the
% corrects being zero
% corrects has the same format as errors
% totalsPerClass has the same format az numErrs
cCalced = squeeze(cCalced);
trueC = squeeze(trueC);
errors = (cCalced ~= trueC).*trueC;
corrects = (cCalced == trueC).*trueC;
classes = 1:10;
errorsPerClass = bsxfun(@eq,errors,classes);
correctsMat = bsxfun(@eq,corrects,classes);
totalsPerClass = bsxfun(@eq,trueC,classes);
totalsPerClass = sum(totalsPerClass);
numErrsPerClass = sum(errorsPerClass,1);


end
function [NegNum] = calcNegNumPerClass(errNum,totals,RateType)

corrNum = totals - errNum;
switch RateType
    case 'double'
    NegNum = min(errNum*2,corrNum);
    case '4'
    NegNum = min(errNum*4,corrNum);
    case 'equal'   
    NegNum = min(errNum,corrNum);
    case 'half'
    NegNum = ceil(min(errNum,corrNum)/2);
    otherwise
        error('unknown');
end
    
end
function Inds = calcNegInds(Loss,correctsMat,NegNum,choiceType)
% NegNum is 1*C
% corrects is n*C with ones 
% Loss is 1,1,1,n
% 
    
    
    switch choiceType
        case 'less'
            Loss = squeeze(Loss);
            LossMat = bsxfun(@times,Loss,correctsMat);
            LossMatSent = LossMat;
            LossMatSent(~correctsMat(:)) = inf;
            [M,I] = sort(LossMatSent,1);
            IndsMat = indicesOfCol(size(I),NegNum);
            Inds = I(IndsMat);
        case 'more'    
            Loss = squeeze(-Loss);
            LossMat = bsxfun(@times,Loss,correctsMat);
            LossMatSent = LossMat;
            LossMatSent(~correctsMat(:)) = inf;
            [M,I] = sort(LossMatSent,1);
            IndsMat = indicesOfCol(size(I),NegNum);
            Inds = I(IndsMat);
        case 'random'
            Loss = squeeze(0*Loss);
            LossMat = bsxfun(@times,Loss,correctsMat);
            LossMatSent = LossMat;
            LossMatSent(~correctsMat(:)) = inf;
            [M,I] = sort(LossMatSent,1);
            IndsMat = indicesOfCol(size(I),NegNum);
            Inds = I(IndsMat);
        otherwise
            error('unknown');
    end
end



%% distLabel funcs
function [c,Disturbed] = distLabel(x,c,Rate)
Directed = true;
[M,I] =   max(x,[],3);
I = single(gather(I));
Disturbed = (I~=c);
if Directed
    Disturbed_2 = rand(size(c),'single');
    Disturbed_2 = Disturbed_2<Rate;
    Disturbed = Disturbed & Disturbed_2;
    c(Disturbed) = I(Disturbed);
    
else
Disturbed = rand(size(c),'single');
Disturbed = Disturbed<Rate;
if sum(Disturbed(:)) ~=0 
c_rand = generateLabels(size(c),10);
c(Disturbed) = c_rand(Disturbed);
end
end
end
function c = generateLabels(SZ,Max)
c = rand(SZ)*Max;
c(c == Max) = Max-1;
c = floor(c);
end