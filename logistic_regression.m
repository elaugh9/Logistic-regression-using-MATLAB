%% Tutorial#5 - Implement Logistic Regression
% Clear all variables and close all plots
clear all; close all; 

%% ******************* Loading Data **********************
data = load('dataset.data');
disp('The dataset was loaded sucessfully!');

X = data(:,1:end-1);% features
y = data(:,end);% class labels

disp('Press any key to continue.');
pause;

%% ****************** Calculate Cost and Gradient *******************
X = [ones(size(X,1),1) , X];
[cost, grad] = calculateCost(zeros(size(X,2),1), X, y);
disp('The cost when theta values initialized to zeros');
disp(cost);

%% ******************* Learn Theta Values *******************
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  This function will return the learned theta values and the cost 
[theta, cost] = ...
	fminunc(@(t)(calculateCost(t, X, y)), zeros(size(X,2),1), options);

disp('Theta values found by fminunc');
disp (theta);