function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


%Add 1s to each layers by using the [ones(1, size(a(layer_num),2));
%a(layer_num)]

a1 = X';
a2 = sigmoid(Theta1 * [ones(1, size(a1, 2)); a1]);
a3 = sigmoid(Theta2 * [ones(1, size(a2, 2)); a2]);

%Retrieve the indices as the prediction value
[u,v] = max(a3);

p = v';




end
