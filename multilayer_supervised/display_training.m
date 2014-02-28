function display_training( iterates, loss_func_val, test_error, train_error, options )
%DISPLAY_TRAINING display training
%   input parameters:
%       iterates : vector;
%       loss_func_val : vector, the value of loss function after each
%       iterate;
%       train_error : vector, the train error after each iterate;
%       test_error : vector, the test error after each iterate;
%       options : struct, contains other information below
%           save : struct
%               filename : string
%               format : string

formats = cell(1,2);
formats{1} = 'jpg';
formats{2} = 'fig';

h = figure;
plot(iterates, loss_func_val, 'LineWidth',2);
title('loss function value');
xlabel('iterates');
ylabel('loss_func_val');
if (exist('options','var') && isfield(options,'save'))
    for i = 1 : size(options.save.format,2)
        saveas(h,[options.save.filename '_Lfv'],formats{options.save.format(i)});
    end
    close(h);
end

h = figure;
plot(iterates,train_error,iterates,test_error,'LineWidth',2);
title('training');
xlabel('iterates');
ylabel('error');
legend('trainError','testError');
if (exist('options','var') && isfield(options,'save'))
    for i = 1 : size(options.save.format,2)
        saveas(h,[options.save.filename '_Err'],formats{options.save.format(i)});
    end
    close(h);
end

end
