function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% calculate cost function
diff = X*theta - y;

NewTheta = [0 ; theta(2:end, :)]; %exclude theta(1) in regularization

J = (diff'*diff)/(2*m) + lambda*(NewTheta'*NewTheta)/(2*m);

% calculate grads
grad = (X'*diff+lambda*NewTheta)/m;

grad = grad(:);

end
