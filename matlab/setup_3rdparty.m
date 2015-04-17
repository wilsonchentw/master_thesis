function setup_3rdparty()
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));

    % Add SPAMS(SPArse Modeling Software) path
    addpath('../spams/spams-matlab/test_release');
    addpath('../spams/spams-matlab/src_release');
    addpath('../spams/spams-matlab/build');
end


