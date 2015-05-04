function [J, grad] = costFunctionReg(theta, X, y, lambda, num_labels=2)
	%COSTFUNCTION Compute cost and gradient for logistic regression, Multi-class classification available if num_labels > 2;

	m = length(y);
	h = sigmoid(X*theta); 

	% Without Regularization
	J = 1.0/m * sum( (-y) .* log(h) - (1.- y) .* log(1.-h) );
	grad = 1.0/m .* ( X' * (h .- y ) ); %'
	%grad_w_reg = 1.0/m .* ( (h.-y') * X ); %'	

	% With regularization (note: Theta0 should be ignored with Regularization)
	theta1 = [0 ; theta(2:size(theta), :)];
	
	J = J + lambda/(2*m) .* sum(theta1 .^2);
	grad = grad .+ (1.0/m) .* (lambda .* theta1); 

	% For fmincg func in one-vs-all:
	grad = grad(:);
end