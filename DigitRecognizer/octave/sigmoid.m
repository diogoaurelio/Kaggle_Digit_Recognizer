%sigmoid.m %

function g = sigmoid(z)
	% Computes the sigmoid function of a given z
	g = 1.0 ./(1.0 + exp(-z));
end