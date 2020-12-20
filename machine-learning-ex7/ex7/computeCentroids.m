function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%Loop over each centroid
% for k = 1:K
	% %Get a vector of 1s and 0s which shows which training example
	% %belongs to centroid k
	% cent_mask = idx == k;
	% %Get the number of training examples assigned to centroid k
	% num_examples = sum(cent_mask);
	% %Convert cent_mask into matrix with same number of columns as X
	% cent_mask = repmat( cent_mask, 1, n );
	% %Multiply cent_mask elementwise with X to get only the examples assigned
	% % to centroid k
	% X_k = X.*cent_mask;
	% %Get the average of the training examples
	% cent_average = (1/num_examples)*sum(X_k,1);
	% %Assign Kth centroid to cent_average
	% centroids(k,:) = cent_average;
% end

%Alternate implementation using find
%Loop over all the centroids
for k = 1:K
	%Get a vector of 1s and 0s which shows which training example
	%belongs to centroid k
	cent_mask = idx == k;
	%Find all the non-zero indices
	non_zero_indices = find(cent_mask);
	%Get all the rows of X corresponding to non-zero indices
	X_k = X(non_zero_indices,:);
	%Get number of training examples
	m_new = size(X_k,1);
	%Get the average of the training examples
	centroids(k,:) = (1/m_new)*(sum(X_k,1));
end

% =============================================================


end

