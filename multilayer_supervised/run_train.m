function run_train(ei,options,data_train, labels_train, data_test, labels_test, display_str, data_path, figure_path)
fprintf(['***** ' display_str ' *****\n']); 
for i = 1 : size(ei,2);
    fprintf([display_str '_' int2str(i) '\n']);
    ei(i)
    options
    %% setup random initial weights
    stack = initialize_weights(ei(i));
    params = stack2params(stack);
    %% check gradient computing
    %grad_error_rate = grad_check(@supervised_dnn_cost, params, 20, ei, data_train, labels_train);
    %fprintf('gradient computing error rate: %f\n',grad_error_rate);

    %% run training
    tic;
    [opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
        params,options,ei(i), data_train, labels_train);
    time_used = toc;
    fprintf(output.message);
    fprintf('\ntime used: %f\n',toc);

    %% display training
    acc_test = zeros(size(output.trace.fval));
    acc_train = acc_test;
    iterates = 1 : size(output.trace.fval,1);
    for iter = iterates
        [~, ~, pred] = supervised_dnn_cost( output.trace.xs(iter,:)', ei(i), data_test, [], true);
        [~,pred] = max(pred);
        acc_test(iter) = mean(pred'==labels_test);

        [~, ~, pred] = supervised_dnn_cost( output.trace.xs(iter,:)', ei(i), data_train, [], true);
        [~,pred] = max(pred);
        acc_train(iter) = mean(pred'==labels_train);
    end
    fprintf('train accuracy : %f\n', acc_train(end));
    fprintf('test accuracy : %f\n', acc_test(end));

    options_display.save.filename = [figure_path '\\' display_str '_' int2str(i)];
    options_display.save.format = [1 2];
    display_training(iterates, output.trace.fval, 1 - acc_test, 1 - acc_train, options_display);

    filename = [data_path '\\' display_str '_' int2str(i)];
    save(filename,'ei','options','options_display','output','time_used','opt_params','opt_value','exitflag','acc_train','acc_test','iterates');

end
end