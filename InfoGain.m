% compute information gain using entropy
function [splitVal, gain] = InfoGain(x, y, weight)

   N = length(x);
   gainMat = zeros(N,1);
   % compute entropy of root
   p = sum(logical(y==1).*weight);
   parent = computeB(p);
   % try each feature value as split point
   
   for i = 1:N 
     index_more = logical(x >= x(i));
     % compute entropy for left and right child
     num1 = sum(weight(index_more));
     num2 = 1 - num1;
     p1 = sum(logical(y(index_more) == 1).*weight(index_more));
     p2 = sum(logical(y(~index_more) == 1).*weight(~index_more));
     entropy1 = computeB(p1./num1).*num1;
     entropy2 = computeB(p2./num2).*num2;
     % compute information gain
     gainMat(i) = parent - (entropy1 + entropy2);
   end
   
   % find max information gain and corresponding split point
   [gain,splitIndex] = max(gainMat);
   splitVal = x(splitIndex);
end

