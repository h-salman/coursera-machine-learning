function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%Vector containing possible values for C	
C_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sig_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

err_min = inf;

%Loop over each value of C and sigma
%Compare with the minimum prediction error
for i=1:size(C_values,1)
	for j=1:size(sig_values,1)
		%Train the model using the chosen values of C and sigma
		model = svmTrain(X, y, C_values(i), @(x1,x2) gaussianKernel(x1, x2, sig_values(j)));
		%Get predictions using the trained model and the validation set
		pred= svmPredict(model, Xval);
		%Get the prediction error
		pred_err = mean(double(pred ~=yval));
		
		if (pred_err < err_min)
			C = C_values(i);
			sigma = sig_values(j);
			err_min = pred_err;
		end
	end
end

fprintf('best C = %f \nbest sigma = %f\nsmallest error = %f\n', C, sigma, err_min)

% =========================================================================

end
