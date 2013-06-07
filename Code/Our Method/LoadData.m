function [ binned_signal ] = LoadData(  )

%% Initialization
f_dir = '..\..\Data\raster\';
f_name = 'bp1001spk_01A_raster_data';
f_path = [f_dir f_name '.mat'];
m_signal = load(f_path);
n_cv_split = 10;

m_interval = 50;
m_signallength = size(m_signal.raster_data,2);

for m_width = 150:50:m_signallength-150
    %% Load data
    n_bin = (m_signallength-m_width)/m_interval+1;
    savefilename = create_binned_data_from_raster_data(f_dir,f_name,50*3,50);
    
    %% Creating a classifier and a preprocessor
    the_feature_preprocessors{1} = zscore_normalize_FP;
    the_classifier = max_correlation_coefficient_CL;
    
    %% Generalize Datasourse
    id_string_names = {'car', 'couch', 'face', 'kiwi', 'flower', 'guitar', 'hand'};
    
    for iID = 1:7
        the_training_label_names{iID} = {[id_string_names{iID} '_upper']};
        the_test_label_names{iID} = {[id_string_names{iID} '_lower']};
    end
    
    specific_labels_names_to_use = 'combined_ID_position';  % use the combined ID and position labels
    ds = generalization_DS(savefilename, specific_labels_names_to_use, n_cv_split, the_training_label_names, the_test_label_names);
    
    the_cross_validator =  standard_resample_CV(ds, the_classifier, the_feature_preprocessors);
    
    % run the decoding analysis
    the_cross_validator = standard_resample_CV(ds, the_classifier, the_feature_preprocessors);
    the_cross_validator.num_resample_runs = 10;
    DECODING_RESULTS = the_cross_validator.run_cv_decoding;
    
    save_file_name = ['position_invariance_results/Zhang_Desimone_pos_inv_results_train_pos' num2str(iTrainPosition) '_test_pos' num2str(iTestPosition)]
    
    save(save_file_name, 'DECODING_RESULTS')
    
    R = DECODING_RESULTS.ZERO_ONE_LOSS_RESULTS.mean_decoding_results;
end



end

