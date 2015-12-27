function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


H = sigmoid(X*theta);

theta0 = [0; theta(2:end)];

J = (1/m)*(-y'* log(H) - (1 - y)'*log(1-H))+(lambda/(2*m))*theta0'*theta0;

grad = (X'*(H-y))/m + lambda/m * theta0;



grad = grad(:);

end
