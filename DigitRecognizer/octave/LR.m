%% LR.m
%% This is simple implementation of one-vs-all;

%% Initialization
clear ; close all; clc

fprintf('\nApplying Logistic Regression for Multi-class classification one-vs-all :\n');

train_csv = './../train.csv';
train = csvread(train_csv);
%X = train(:,2:end); % alias
%y = train(:,1);

% Just for the local PC testing
X = train(1:10, 2:end);
y = train(1:10,1);


% m - sample size
% n - num. of features
[m, n] = size(X);


% Adding intercept
X_w_inter = [ones(m,1) X];

initial_theta = zeros(n+1, 1);

%% ================ Part 1: Compute and display initial cost and gradient ================ 
%[cost, grad] = costFunction(initial_theta, X_w_inter, y);


%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);
% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0.1;
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that "0" is mapped to label 10)

%  Run fminunc to obtain the optimal theta: 
% https://www.gnu.org/software/octave/doc/interpreter/Minimizers.html
%[theta, cost] = ...
%	fminunc(@(t)(costFunction(t, X_w_inter, y)), initial_theta, options);

% with Regularization:
%[theta, cost] = ...
%	fminunc(@(t)(costFunctionReg(t, X_w_inter, y, lambda)), initial_theta, options);

% Print theta to screen
% fprintf('Cost at theta found by fminunc: %f\n', cost);
% fprintf('theta: \n');
% fprintf(' %f \n', theta);

%[cost, grad] = costFunctionReg(initial_theta, X_w_inter, y, lambda, num_labels);

%thetaOneVsAll = zeros(num_labels, 1 + n);
% One-VS-All approach:
	% a) compute the cost for each class
	% b) choose Theta parameters where cost is minimum;

for c=1:num_labels
	thetaOneVsAll(c,:) = ...
	         fmincg (@(t)(costFunctionReg(t, X_w_inter, (y == c), lambda)), ...
	                 initial_theta, options);
end

%for c=1:num_labels
%	thetaOneVsAll(c,:) = ...
%	         fminunc (@(t)(costFunctionReg(t, X_w_inter, (y == c), lambda)), ...
%	                 initial_theta, options);
%end

%% ================ Part 2: Predict for One-Vs-All ================

prediction = predictOneVsAll(thetaOneVsAll, X);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(prediction == y)) * 100);

fprintf('\nUsing Test Set :\n');

%% ================ CV/test set ================
%rand_indices = randperm(m);
%a = rand_indices(1:20);
%X_cv = train(a, 2:end);
%y_cv = train(a, 1);

%prediction_cv = predictOneVsAll(thetaOneVsAll, X_cv);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(prediction_cv == y_cv)) * 100);


test_csv = './../test.csv';
X_test = csvread(test_csv);
y_test = X_test(:, 1); % y_test(1:10,:)
prediction_test = predictOneVsAll(thetaOneVsAll, X_test);

y_submit = [ 1 : 1 : size(y_test,1)];
%final = [y_test, prediction_test']; %'
final = [y_submit', prediction_test']; %

% ToDo: correct output csv for Kaggle format
csvwrite ("lrKaggleDigitRecogv.csv", final);






