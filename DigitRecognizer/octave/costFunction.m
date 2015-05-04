function [J, grad] = costFunction(theta, X, y)
	%COSTFUNCTION Compute cost and gradient for logistic regression
	m = length(y);
	h = sigmoid(X*theta);
	%J = 1.0/m * sum( -y * log(h') - (1.-y) * log(1.-h') ); 
	J = 1.0/m * ( log(h) * (-y) - log(1.-h) * (1.- y) );
	grad = 1.0 /m .* ( (h.-y') * X ); 
end