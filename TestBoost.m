% script to graphically report the training and test errors
% as a function of the number of weak hypothesis

% load train and test data
load zip.test;
ziptest = zip;
load zip.train;
% one-vs-three problem
fprintf('Working on the one-vs-three problem...');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
% label digit 1 as +1 and digit 3 as -1
y_tr = subsample(:,1);
idx = logical(y_tr==1);
y_tr(idx) = 1;
y_tr(~idx) = -1;
X_tr = subsample(:,2:257);
% generate test set
testsubsample1 = ziptest(find(ziptest(:,1)==1 | ziptest(:,1) == 3),:);
y_te1 = testsubsample1(:,1);
idx = logical(y_te1==1);
y_te1(idx) = 1;
y_te1(~idx) = -1;
X_te1 = testsubsample1(:,2:257);
% compute errors for different number of weak hypothesis
numIter = 200;
num_hyp = [1:1:numIter];
[train_err1,test_err1] = AdaBoost(X_tr,y_tr,X_te1,y_te1,numIter);
% make plot
figure
plot(num_hyp,train_err1,'-ro');
title('One-vs-Three Problem');
xlabel('Number of weak hypothesis');
ylabel('Training Error');
figure
plot(num_hyp,test_err1,'-bx');
title('One-vs-Three Problem');
xlabel('Number of weak hypothesis');
ylabel('Test Error');
figure
plot(num_hyp,train_err1,'-ro',num_hyp,test_err1,'-bx');
title('One-vs-Three Problem');
xlabel('Number of weak hypothesis');
ylabel('Error');
legend('Training error','Test error');
%%

% three vs. five problem
fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
% label digit 3 as +1 and digit 5 as -1
y_tr = subsample(:,1);
idx = logical(y_tr==3);
y_tr(idx) = 1;
y_tr(~idx) = -1;
X_tr = subsample(:,2:257);
% generate test set
testsubsample2 = ziptest(find(ziptest(:,3)==3 | ziptest(:,1) == 5),:);
y_te2 = testsubsample2(:,1);
idx = logical(y_te2==3);
y_te2(idx) = 1;
y_te2(~idx) = -1;
X_te2 = testsubsample2(:,2:257);
% compute errors for different number of weak hypothesis
[train_err2,test_err2] = AdaBoost(X_tr,y_tr,X_te2,y_te2,numIter);
% make plot
figure
plot(num_hyp,train_err2,'-ro');
title('Three-vs-Five Problem');
xlabel('Number of weak hypothesis');
ylabel('Training Error');
figure
plot(num_hyp,test_err2,'-bx');
title('Three-vs-Five Problem');
xlabel('Number of weak hypothesis');
ylabel('Test Error');
figure
plot(num_hyp,train_err2,'-ro',num_hyp,test_err2,'-bx');
title('Three-vs-Five Problem');
xlabel('Number of weak hypothesis');
ylabel('Error');
legend('Training error','Test error');