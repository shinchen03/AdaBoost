% predict and compute classification errors using decision stump
%   alpha: alpha value associated witht the learner
%   stp: decision stump
%   i: number of weak learners
%   X: feature matrix
%   y: labels

function [err] = boostPredict(alpha,weakLearners,i,X,y)
    % number of examples
    N = size(X,1);
    % matrix to store predictions
    m = zeros(N,i);
    % each column corresponds to predictions 
    % using k rounds
    for k = 1:i
        m(:,k) = alpha(k).*weakLearners{k}.predict(X);
    end
    % compute current hypothesis
    m = sum(m,2);
    % label +1/-1 based on sign
    label = sign(m);
    % compute error
    err_label = logical(label ~= y);
    err = sum(err_label)./N;
end

