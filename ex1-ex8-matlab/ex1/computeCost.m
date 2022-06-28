function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%循环做法
% hypothesis=0;
% n=size(X,2);%训练集的特征维度
% for   i=1:m
%     for j=1:n
%          hypothesis=hypothesis+(theta(j)*X(i,j));
%     end
%     J=J+(hypothesis-y(i))^2;
% end
% J=J/(2*m);

%矩阵(向量做法)
hypothesis=X*theta;%hypothesis值(列向量)
sqrterror=(hypothesis-y).^2;%误差列向量
J=1/(2*m)*sum(sqrterror);

% =========================================================================

end
