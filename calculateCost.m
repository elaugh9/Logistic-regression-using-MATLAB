%% The function compute the cost and gradient of a logistic regression model 
% Parameters:
% X is the matrix of the features for all the examples (m)
% y is the vector of the targets for all the examples (m)
% theta is a column vector of all theta values

%% The function compute the cost and gradient of a logistic regression model 
% Parameters:
% X is the matrix of the features for all the examples (m)
% y is the vector of the targets for all the examples (m)
% theta is a column vector of all theta values
function [J, grad] = costFunction(theta, X, y)

m = size(X, 1);
h = 1 ./ (1 + exp(-X * theta));  
J = - (1 / m) * sum( y .* log(h) + (1 - y) .* log(1 - h));

grad = zeros(size(theta,1), 1);
for i = 1 : size(grad)
grad(i) = (1 / m) * sum( (h - y)' * X(:, i));

end
