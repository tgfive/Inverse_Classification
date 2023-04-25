python3 -m invclass.res_plot --data_path ./brazil_data/ --data_file processed_A757_inv11-15.pkl \
	--result_file processed_A757_inv11-15-invResult.pkl --image_path ./documentation/A757_weather_ind11-15/ \
	--util_file A757_weather.csv

python3 -m invclass.data_plot --data_path ./brazil_data/ --data_file processed_A757_inv11-15.pkl \
	--result_file processed_A757_inv11-15-invResult.pkl --image_path ./documentation/A757_weather_ind11-15/ \
	--util_file A757_weather.csv
