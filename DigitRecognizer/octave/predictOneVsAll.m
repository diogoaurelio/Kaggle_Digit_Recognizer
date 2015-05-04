function p = predictOneVsAll(thetaOneVsAll, X)
	m = size(X,1);
	num_labels = size(thetaOneVsAll,1);
	X = [ones(m,1) X];
	% note: thetaOneVsAll = zeros(num_labels, 1 + n)
	
	%[x, p] = max( sigmoid( X * thetaOneVsAll' )); %'
	
	for i=1:m,
		[x, p(i)] = max( sigmoid(X(i,:) * thetaOneVsAll') ); %'
	end
end