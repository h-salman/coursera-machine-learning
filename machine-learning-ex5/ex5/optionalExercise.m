%% Machine Learning online class
% Optional ungraded Exercise

%% Initialization
clear ; close all; clc

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

p=8;
m = size(y);
% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];    

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

%Choose a value for lambda
lambda = 0.01;

%Create a vector with different num samples
%Ranging from 1 percent of the dataset to 100% of the dataset

num_samples = zeros(10,1);

for i=1:10
	num_samples(i) = floor((i/10)*size(X,1)); %take 80% of the data as num_samples
end

num_iter = 50;

error_train = zeros(10,1);
error_val = zeros(10,1);
%Get the average error over the training set and validation set
for i=1:10
	[error_train(i), error_val(i)] = ...
		randomSampleValidation(X_poly, y, X_poly_val, yval, lambda, ...
		num_samples(i), num_iter);
end

%Plot the curves
figure(1);
plot(num_samples, error_train, num_samples, error_val);
xlabel('Number of training examples')
ylabel('Error')
legend('Train', 'Cross Validation')