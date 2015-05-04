%% Neural Network in Octave/Matlab

%% Disclamer: this solution is based on the very good materials 
%% provided by Andrew Ng, in his course Machine Learning Course, 
%% from Stanford University, available in Coursera.

%% Initialization

%clear ; close all; clc

%% Part 1 - Visualization

% Load Training Data
fprintf('Loading and Visualizing Data...\n')



%load('train.csv');
train_csv = './../train.csv';
train = csvread(train_csv);
%X = train(:,2:end); % alias
%y = train(:,1);

% Just for the local PC testing
X = train(1:5, 2:end);
y = train(1:5,1);
[m, n] = size(X);

fprintf('Data loaded\n');

% Randomly select 100 data points to display
%sel = randperm(size(X,1));
%sel = sel(1:100);
%displayData(X(sel,:));

%fprintf('Paused. Press enter to continue.\n');
%pause;

%% Setup parameters for NN
input_layer_size = 784; % 28 x 28 pixels in height and in width;
hidden_layer_size = 25;
num_labels = 10;

%% NN Cost Function:

%% Using Backpropagation Algorithm to compute the Gradient of the NN Cost function

% 1) Initialize Theta parameters




% a) Random initialize thetas

%%===== Function to random initialize Thetas for Backpropagation algorithm; %%=====
function [thetas] = randomInitialize(thetas) 
	%% Takes in a vector and random initializes values between -1 and 1 
	%% throughout the whole vector

	init_epsilon = 1;
	for i=1:length(thetas)
		thetas(i) = rand(1) * (2*init_epsilon)-init_epsilon;
	end
end

%initial_thetas = randomInitialize(initial_thetas);

%% Or use Logistic Regression to initialize thetas - NOTE: computationaly expensive

function [h] = sigmoid(z)
	h = 1.0/(1+ exp(-z));
end

function [J, gradient] = LRCostFunc4FminUnc(theta, X, y, lambda = 0.01)
	% Computes cost and gradient for logistic regression with regularization
	
	m = size(X,1);
	% Add intercept to X:
	X = [ ones(m,1) X];
	
	
	J = 0;
	h = sigmoid(X * theta);

	regularization = lambda/(2.0*m) .* sum(theta.^2)
	J = (-1.0/m * sum( y .* log(h) .+ (1.-y).*log(1.-h) )) + regularization

	gradient = zeros(size(theta));
	size(gradient)

	%gradient(1) = 1.0/m * sum( (h .- y).*X(:,1)  );
	theta(1) = 0;
	for i=1:size(theta)
		gradReg = lambda/m * theta(i);
		gradient(i) = 1.0/m * sum( (h' .- y).*X(:,i)  ) + gradReg; %'
	end
end

function [J] = LRCostFunc(theta, X, y, lambda = 0.01)
	% Computes cost and gradient for logistic regression with regularization
	
	m = size(X,1);
	% Add intercept to X:
	X = [ ones(m,1) X];
	
	
	J = 0;
	h = sigmoid(X * theta);

	regularization = lambda/(2.0*m) .* sum(theta.^2)
	J = (-1.0/m * sum( y .* log(h) .+ (1.-y).*log(1.-h) )) + regularization

	gradient = zeros(size(theta));
	size(gradient)

	%gradient(1) = 1.0/m * sum( (h .- y).*X(:,1)  );
	theta(1) = 0;
	for i=1:size(theta)
		gradReg = lambda/m * theta(i);
		gradient(i) = 1.0/m * sum( (h' .- y).*X(:,i)  ) + gradReg; %'
	end
end



%% Without Regularization to start:
lambda = 0; 
initial_thetas = zeros(n+1,1);
[cost, grad] = LRCostFunc(initial_thetas, X, y, lambda);

%% Using fminunc function to obtain Initial Theta Parameters:

%  Set options for fminunc / fmincg
options = optimset('GradObj', 'on', 'MaxIter', 3);

% For each of the classes (c):
function [all_theta] = oneVsAll(X, y, initial_thetas, options, num_labels, lambda)
	for c=1:num_labels,	
		%[theta] = ...
		all_theta(c,:) = ...
		         fmincg (@(t)(LRCostFunc(t, X, (y == c), lambda)), ...
		                 initial_thetas, options);
	end
end

[theta] = oneVsAll(X, y, initial_thetas, options, num_labels, lambda);

% Print theta to screen
%fprintf('Cost at theta found by fminunc: %f\n', cost);
%fprintf('theta: \n');
%fprintf(' %f \n', theta);
%pause;


%% Test Predictions
predictions = zeros(m,1);
csvwrite("predictions.csv", predictions);

