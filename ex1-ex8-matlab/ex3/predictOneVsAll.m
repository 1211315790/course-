function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);%分类器数量

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);%m×1向量，每一行表明预测的是第几个分类

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
[row p] = max(X * all_theta',[],2);%X*all_theta',m×n,n列代表n个分类器,m行代表m个参数
%每一行中最大的值表示这个样本属于那个分类器的概率最大,row为m×1,每一行存储最大值,p是m×1存储每一行最大值的索引即所在的分类器也就是预测的值,
%若p与实际值y完全相等则预测全部准确







% =========================================================================


end
