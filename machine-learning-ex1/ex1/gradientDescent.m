function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

fprintf('test begin val of theta:')
theta

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % the cost func is J(theta) = 1/(2m) * sum of (yi - theta1 * xi - theta0) ^ 2
    % d(J(theta0)) wrt theta0 = (1/m) * sum of (yi - theta1 * xi - theta0) * (-1) by chain rule
    % d(J(theta1)) wrt theta1 = (1/m) * sum of [(yi - theta1 * xi - theta0) * (-xi)] by chain rule

    dJ0 = (1/length(y)) * (-1) * sum(y - X*theta);    
    x_hat = X(1:length(X),2);
    dJ1 = (1/length(y)) * (-1) * sum(x_hat.*(y - X*theta));

    new_b0 = theta(1) - alpha * dJ0;
    new_b1 = theta(2) - alpha * dJ1;

    theta = [new_b0; new_b1];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
