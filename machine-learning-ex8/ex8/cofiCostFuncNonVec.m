function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

J = (1/2)*sum( sum( ( ( X*Theta' - Y ).^2 ).*R ) ) + ...
	(lambda/2)*sum( sum( Theta.^2 ) ) + (lambda/2)*sum( sum (X.^2) );

%Calculate the gradient non-vectorized
%Loop over the movies 
%First calculate X_grad

%X is the matrix of movies
%So we loop first over the movies
for i=1:num_movies
	%Secondly, we will loop over the number of users
	%First we will find all the users that have given a ratings
	idx = find( R(i, :) == 1 );
	%Get the relevant rows of theta where R(i,j) == 1
	Theta_temp = Theta(idx, :);
	%Get relevant columns of Y for the ith row
	Y_temp = Y(i, idx);
	%Get the Gradient for the ith movie
	X_grad(i, :) = ( X(i, :)*Theta_temp' - Y_temp )*Theta_temp;
end	
%Loop over the number of users to find the gradient of Theta
for j=1:num_users
	%Loop over the number of movies
	%Find all the movies that have been given a rating by the jth user
	idx = find( R(:,j) == 1);
	%Get the relevant rows of movies that have ratings
	X_temp = X(idx,:);
	%Get the relevant row of Y for the jth column
	Y_temp = Y(idx, j);
	Theta_grad(j, :) = ( Theta(j, :)*X_temp' - Y_temp' )*X_temp; 
end
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
