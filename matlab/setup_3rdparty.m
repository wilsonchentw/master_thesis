function setup_3rdparty()
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));
    %addpath(fullfile('../liblinear-weights/matlab'));
    %addpath(fullfile('../libsvm-weights/matlab'));

    % Add SPAMS(SPArse Modeling Software) path
    addpath(fullfile('../spams/spams-matlab/test_release'));
    addpath(fullfile('../spams/spams-matlab/src_release'));
    addpath(fullfile('../spams/spams-matlab/build'));
end
