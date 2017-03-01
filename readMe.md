11/16/2016 更新：
	修复CPU模式下 valid 参数执行时会报double free的错误；
	valid模式下，在输出信息中添加平均准确率 avg acc
11/19/2016 更新：
	打印出每次从train 数据中读取数据的位置，即args.index值。
11/22/2016 更新：
	增加训练模式选择：
		RANDOM_TRAIN_INDEX：训练数据随机读取；
		else：顺序读取
