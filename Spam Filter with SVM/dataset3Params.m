function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. 

C = 1;
sigma = 0.3;

vec = [ 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

C_vec = vec
S_vec = vec

optima = [1 0 0];
trialNum = 1;

for tryC = C_vec;
    for tryS = S_vec;
        fprintf(['Attempt #%d: C = %f, sigma = %f\n'], trialNum, tryC, tryS);
        trialNum = trialNum + 1;
        model = svmTrain(X, y, tryC, @(x1, x2) gaussianKernel(x1, x2, tryS));
        predictions = svmPredict(model, Xval);
        predError = 1 - mean(double(predictions == yval));
        if (predError < optima(1))
            fprintf(['Best so far %f\n'], predError);
            optima = [predError, tryC, tryS];
        end
    end
end

fprintf(['\n Best Found: C = %f, sigma = %f\n'], optima(2), optima(3));

C = optima(2);
sigma = optima(3);

end
