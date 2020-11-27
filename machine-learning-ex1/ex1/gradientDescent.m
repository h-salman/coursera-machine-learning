function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	predictions = X*theta;
	errors = predictions-y;

	%Calculate new theta parameters
	theta_0_new = theta(1) - alpha*(1/m) * sum(errors);  %for theta_0 no need to multiply by 1st column of X as it is already 1
	theta_1_new = theta(2) - alpha*(1/m) * sum(errors.*X(:,2));
	
	%simultaneous updates
	theta(1) = theta_0_new;
	theta(2) = theta_1_new;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
