% runs experiments procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

system('powercfg -h off');

clc;
clear;

%% output redirection
diary on;

%% saving path
data_path = 'Outcome\\data';
figure_path = 'Outcome\\figure';

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% setup minfunc options
options = [];
options.display = 'off';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

lambdas = {0,0.00005,0.0001,0.00015,0.0002,0.0005,0.001};
% ei = struct('input_dim',784,'output_dim',10,'layer_sizes',[128 64 10],'lambda',lambdas,'activation_fun','logistic');
% 
% display_str = 'exp2';
% run_train(ei,options,data_train, labels_train, data_test, labels_test, display_str,data_path,figure_path);

ei = struct('input_dim',784,'output_dim',10,'layer_sizes',[800 10],'lambda',lambdas,'activation_fun','logistic');
display_str = 'exp4';
run_train(ei,options,data_train, labels_train, data_test, labels_test, display_str,data_path,figure_path);

ei = struct('input_dim',784,'output_dim',10,'layer_sizes',[800 180 10],'lambda',lambdas,'activation_fun','logistic');
display_str = 'exp5';
run_train(ei,options,data_train, labels_train, data_test, labels_test, display_str,data_path,figure_path);

ei = struct('input_dim',784,'output_dim',10,'layer_sizes',[800 200 50 10],'lambda',lambdas,'activation_fun','logistic');
display_str = 'exp6';
run_train(ei,options,data_train, labels_train, data_test, labels_test, display_str,data_path,figure_path);

diary('Outcome\\dnn_log.txt');
diary off;
system('powercfg -h on');
system('shutdown -h');

