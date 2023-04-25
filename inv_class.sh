python3 -m invclass.make_pkl --data_path ./brazil_data/ --data_file A757_weather.csv \
        --util_file brazil_indices.csv --save_file processed_A757_inv11-15.pkl

python3 -m invclass.inv_class --data_path ./brazil_data/ --data_file processed_A757_inv11-15.pkl \
	--model_file reg_model.h5 --ind_model_file ind_model.h5

python3 -m invclass.res_plot --data_path ./brazil_data/ --data_file processed_A757_inv11-15.pkl \
	--result_file processed_A757_inv11-15-invResult.pkl --image_path ./documentation/A757_weather_ind11-15/ \
	--util_file A757_weather.csv

python3 -m invclass.data_plot --data_path ./brazil_data/ --data_file processed_A757_inv11-15.pkl \
	--result_file processed_A757_inv11-15-invResult.pkl --image_path ./documentation/A757_weather_ind11-15/ \
	--util_file A757_weather.csv
