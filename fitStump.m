% train a decision stump using weight
%   X: data matrix
%   y: labels
%   weight: weight on each example

function [stp] = fitStump(X,y,weight)
    % number of features
    dim = size(X,2);
    % initialize information gain mamtrix for each stump
    gain = zeros(dim,1);
    split = zeros(dim, 1);
    % compute each decision stump and corresponding gain
    for i = 1:dim
        [split(i), gain(i)] = InfoGain(X(:,i), y, weight);
    end
    % find decision stump with maximum information gain
    [stpGain,index] = max(gain);
    stpThreshold = split(index);
    % construct decision stump
    stp = decisionStump(X(:,index),y,index, weight, stpGain, stpThreshold);  
    pred = stp.predict(X);
    % compute weighted error of decision stump
    err_label = logical(pred ~= y);
    stp.error = sum(err_label.*weight)./sum(weight);
end

