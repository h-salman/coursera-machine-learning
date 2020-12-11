function [error_train, error_val] = ...
	randomSampleValidation(X, y, Xval, yval, lambda, num_samples, num_iter)
	
% RANDOMSAMPLEVALIDATION Choose random samples from X, y , Xval and yVal to
% calculate the error. num_samples number of samples will be chosen randomly
% The average of both errors will be taken from num_iter

%sum useful 
m = size(y);

error_train = 0;
error_val = 0;

for i=1:num_iter
	%Select num_samples randomly from X_poly and X_val
	train_indices = randperm(m, num_samples); 
	X_train = X(train_indices, :);
	y_train = y(train_indices);
	Xval_train = Xval(train_indices, :);
	yval_train = yval(train_indices);
	
	%Get theta by training the model
	theta_train = trainLinearReg(X_train, y_train, lambda);
	%Accumulate the error from the trained theta
	error_train += linearRegCostFunction(X_train, y_train, theta_train, 0); %cost without regularization
	
	%Calculate the cost of the validation set
	error_val += linearRegCostFunction(Xval_train, yval_train, theta_train, 0);
end

%Calculate the average error over all iterations
error_train = error_train/num_iter;
error_val = error_val/num_iter;


end