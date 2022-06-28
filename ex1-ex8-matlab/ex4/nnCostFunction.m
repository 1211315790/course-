function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%正向传播,向量实现
a1=[ones(m,1) X];%m×401.第一列是偏置单元,剩下400列为20×20灰度特征
Z2=a1*Theta1';
a2=sigmoid(Z2);
a2=[ones(m,1) a2];%m×hidden_layer_size+1,每一行是每个样本的神经单元
Z3=a2*Theta2';
a3=sigmoid(Z3);%m×num_labels,每一行是每个样本的输出层神经单元
initial_y=y;
y=zeros(m,num_labels);%m×num_labels,每一行是每个样本的实际值
for i=1:m
    y(i,initial_y(i))=1;
end


%向量实现
J=sum(sum(y.*log(a3)+(1-y).*log(1-a3)));
J=-1/m*J;%未正则化
theta1_sum=sum(sum(Theta1(:,2:end).^2));%theta1所有参数的平方和
theta2_sum=sum(sum(Theta2(:,2:end).^2));
J=J+lambda/(2*m)*(theta1_sum+theta2_sum);%正则化


%反向传播
delta3=a3-y;%m×num_labels,每一行是每个样本的神经单元误差
Theta2_grad=delta3'*a2/m;
delta2=delta3*Theta2.*a2.*(1-a2);
delta2=delta2(:,2:end);
Theta1_grad=delta2'*a1/m;


%正则化神经网络
Theta1_temp = Theta1;
Theta1_temp(:, 1) = 0;
Theta2_temp = Theta2;
Theta2_temp(:, 1) = 0;
Theta2_grad=Theta2_grad+lambda*Theta2_temp/m;
Theta1_grad=Theta1_grad+lambda*Theta1_temp/m;


 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
