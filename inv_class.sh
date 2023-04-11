python3 -m invclass.make_pkl --data_path ./brazil_data/ --data_file A036_weather.csv \
        --util_file brazil_indices.csv --save_file processed_A036_inv.pkl

python3 -m invclass.inv_class --data_path ./brazil_data/ --data_file processed_A036_inv.pkl \
	--model_file reg_model.h5 --ind_model_file ind_model.h5

python3 -m invclass.res_plot --data_path ./brazil_data/ --data_file processed_A036_inv.pkl \
	--result_file processed_A036_inv-invResult.pkl --image_path ./documentation/ \
	--util_file A036_weather.csv
