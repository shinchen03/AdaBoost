function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use

% number of training examples
N_tr = size(X_tr,1);
% initial weight 
weight = repmat(1/N_tr,N_tr,1);
% initialize all weak learners used 
weakLearners = cell(n_trees,1);
% initialize alpha values associated with each learner
alpha = zeros(n_trees,1);
% initialize training and test error for each round
train_err = zeros(n_trees,1);
test_err = zeros(n_trees,1);

for i = 1:n_trees
    % build decision stump using current weight
    weakHypo = fitStump(X_tr, y_tr, weight);
    % store current weak learner
    weakLearners{i} = weakHypo;
    % update alpha
    alpha(i) = 0.5*log((1-weakHypo.error)/weakHypo.error);
    % use weak learner for prediction
    label = weakHypo.predict(X_tr);
    % update weight
    temp = -1*alpha(i)*(y_tr.*label); 
    temp = weight.*exp(temp); 
    weight = temp./sum(temp); 
    % compute training and test error
    train_err(i) = boostPredict(alpha, weakLearners, i, X_tr, y_tr);
    test_err(i) = boostPredict(alpha, weakLearners, i, X_te, y_te);
end

