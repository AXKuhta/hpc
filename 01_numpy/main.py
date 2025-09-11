import pandas as pd
import numpy as np

def do_hanoi():
	#
	# Загрузить датасет через pandas, дальше через numpy
	#
	vietnam = pd.read_csv("vietnam-weather.csv")
	vietnam = np.array(vietnam)

	# Столбцы:
	# 0 province
	# 1 max
	# 2 min
	# 3 wind
	# 4 wind_d
	# 5 rain
	# 6 humidi
	# 7 cloud
	# 8 pressure
	# 9 date
	hanoi = vietnam[vietnam[:, 0] == "Hanoi"]
	hanoi_rain = hanoi[:, 5]

	minor_rain = hanoi[ (hanoi_rain > 1) * (hanoi_rain < 10) ]
	any_rain = hanoi_rain[ hanoi_rain > 0 ]
	mean_rain = np.mean(any_rain)

	mask_2020 = (hanoi[:, 9] >= "2020-01-01") * (hanoi[:, 9] <= "2020-12-31")
	hanoi_2020_mean_rain = np.mean(hanoi_rain[mask_2020])
	hanoi_2020_max_rain = np.max(hanoi_rain[mask_2020])

	hanoi_not_2020_max_rain = np.max(hanoi_rain[~mask_2020])

	print("В Ханое:")
	print("- дней с уровнем осадков менее 10 мм и более 1мм:", len(minor_rain))
	print(f"- среднее количество осадков в дождливые дни: {mean_rain:.1f}мм")
	print(f"- среднее количество осадков в 2020: {hanoi_2020_mean_rain:.1f}мм")
	print(f"- макс. количество осадков в 2020: {hanoi_2020_max_rain:.1f}мм")
	print(f"- макс. количество осадков не в 2020: {hanoi_not_2020_max_rain:.1f}мм")

def do_seattle():
	#
	# 0 date
	# 1 precipitation
	# 2 temp_max
	# 3 temp_min
	# 4 wind
	# 5 weather
	#
	seattle = pd.read_csv("seattle-weather.csv")
	seattle = np.array(seattle)

	seattle_rain = seattle[:, 1]

	minor_rain = seattle_rain[ (seattle_rain > 1) * (seattle_rain < 10) ]
	any_rain = seattle_rain[ seattle_rain > 0 ]
	mean_rain = np.mean(any_rain)

	mask_2014 = (seattle[:, 0] >= "2014-01-01") * (seattle[:, 0] <= "2014-12-31")
	seattle_2014_mean_rain = np.mean(seattle_rain[mask_2014])
	seattle_2014_max_rain = np.max(seattle_rain[mask_2014])

	seattle_not_2014_max_rain = np.max(seattle_rain[np.invert(mask_2014)])

	print("В Сиэтле:")
	print("- дней с уровнем осадков менее 10 мм и более 1мм:", len(minor_rain))
	print(f"- среднее количество осадков в дождливые дни: {mean_rain:.1f}мм")
	print(f"- среднее количество осадков в 2014: {seattle_2014_mean_rain:.1f}мм")
	print(f"- макс. количество осадков в 2014: {seattle_2014_max_rain:.1f}мм")
	print(f"- макс. количество осадков не в 2014: {seattle_not_2014_max_rain:.1f}мм")

def do_warsaw():
	# 0 "STATION"
	# 1 "NAME"
	# 2 "LATITUDE"
	# 3 "LONGITUDE"
	# 4 "ELEVATION"
	# 5 "DATE"
	# 6 "PRCP"
	# 7 "SNWD"
	# 8 "TAVG"
	# 9 "TMAX"
	# 10 "TMIN"
	#
	warsaw = pd.read_csv("warsaw-weather.csv")
	warsaw = np.array(warsaw)

	warsaw_rain = np.float32(warsaw[:, 6])
	warsaw_rain[np.isnan(warsaw_rain)] = 0.0

	minor_rain = warsaw_rain[ (warsaw_rain > 1) * (warsaw_rain < 10) ]
	any_rain = warsaw_rain[ warsaw_rain > 0 ]
	mean_rain = np.mean(any_rain)

	mask_2019 = (warsaw[:, 5] >= "2019-01-01") * (warsaw[:, 5] <= "2019-12-31")
	warsaw_2019_mean_rain = np.mean(warsaw_rain[mask_2019])
	warsaw_2019_max_rain = np.max(warsaw_rain[mask_2019])

	warsaw_not_2019_max_rain = np.max(warsaw_rain[np.invert(mask_2019)])

	print("В Варшаве:")
	print("- дней с уровнем осадков менее 10 мм и более 1мм:", len(minor_rain))
	print(f"- среднее количество осадков в дождливые дни: {mean_rain:.1f}мм")
	print(f"- среднее количество осадков в 2019: {warsaw_2019_mean_rain:.1f}мм")
	print(f"- макс. количество осадков в 2019: {warsaw_2019_max_rain:.1f}мм")
	print(f"- макс. количество осадков не в 2019: {warsaw_not_2019_max_rain:.1f}мм")

do_hanoi()
do_seattle()
do_warsaw()
