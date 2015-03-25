function [h] = sigmoid(z)
	h = 1.0/(1+ exp(-z));
end

function [J, gradient] = LRCostFunc(theta, X, y, lambda = 0.01)
	% Computes cost and gradient for logistic regression with regularization
	
	m = size(X,1);
	J = 0;
	h = sigmoid(X * theta);

	regularization = lambda/(2.0*m) .* sum(theta.^2);
	J = (-1.0/m * sum( y .* log(h) + (1.-y).*log(1.-h) )) + regularization;

	gradient = zeros(size(X,1),1);

	gradient(1) = 1.0/m * sum( (h .- y).*X(:,1)  );
	for i=2:size(gradient,1)
		gradReg = lambda/m * theta(i); 
		gradient(i) = 1.0/m * sum( (h .- y).*X(:,i)  ) + gradReg;  
	end
end

