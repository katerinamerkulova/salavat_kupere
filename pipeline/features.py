feats = [ 
	'pic_dist_0_perc',
	'pic_dist_25_perc',
	'pic_dist_50_perc',
	'fuzzywuzzy_ratio_cleaned_name',
	'fuzzywuzzy_ratio_name',

    ## bert
    'euclidean_name_bert_dist',
	'cosine_name_bert_dist',
	'jensenshannon_name_bert_dist',
	'minkowski_name_bert_dist',
	'sqeuclidean_name_bert_dist',

    ## minilm
	'euclidean_name_minilm_dist',
	'cosine_name_minilm_dist',
	'jensenshannon_name_minilm_dist',
	'minkowski_name_minilm_dist',
	'sqeuclidean_name_minilm_dist',
    
	## minilm name + cat
	# "euclidean_name_cat_minilm_dist", 
	# "cosine_name_cat_minilm_dist",
	# "jensenshannon_name_cat_minilmt_dist",
	# "minkowski_name_cat_minilm_dist",
	# "sqeuclidean_name_cat_minilm_dist",

    ## labse
    # "euclidean_name_labse_dist", 
	# "cosine_name_labse_dist",
	# "jensenshannon_name_labse_dist",
	# "minkowski_name_minilm_dist",
	# "sqeuclidean_name_minilm_dist",

	## rubert
	# "euclidean_name_rubert_dist", 
	# "cosine_name_rubert_dist",
	# "jensenshannon_name_rubert_dist",
	# "minkowski_name_rubert_dist",
	# "sqeuclidean_name_rubert_dist",

    ## color
	'is_same_color',
	'partial_ratio_color',
    'is_same_color_Сетевые фильтры, разветвители и удлинители',
	'is_same_color_Батарейки и аккумуляторы',
	'is_same_color_Смартфоны, планшеты, мобильные телефоны',
	'is_same_color_Расходник для печати',
	'is_same_color_Кабели и переходники',
	'is_same_color_Устройство ручного ввода',
	'is_same_color_Смарт-часы',
	'is_same_color_Аксессуары для фото и видеотехники',
	'is_same_color_SIM-карты',
	'is_same_color_Запчасти для смартфонов',
	'is_same_color_ИБП',
	'is_same_color_Запчасти для ноутбуков',
	'is_same_color_Видеокарты и графические ускорители',
	'is_same_color_Акустика и колонки',
	'is_same_color_Компьютер',
	'is_same_color_Корпуса для компьютеров',
	'is_same_color_Электронное оборудование для торговли',
	'is_same_color_Видеонаблюдение',
	'is_same_color_Зарядные устройства и док-станции',
	'is_same_color_Наушники и гарнитуры',
	'is_same_color_Видеорегистратор',
	'is_same_color_Электронные модули',
	'is_same_color_Кронштейн',
	'is_same_color_Офисная техника',
	'is_same_color_Оптические приборы',
	'is_same_color_Мониторы и запчасти',
	'is_same_color_Жесткие диски, SSD и сетевые накопители',
	'is_same_color_Чехол',
	'is_same_color_Телевизоры',
	'is_same_color_Фотоаппараты',
	'is_same_color_Сетевое оборудование',
	'is_same_color_Принтеры и МФУ',
	'is_same_color_Проводные и DECT-телефоны',
	'is_same_color_Защитные пленки и стекла',
	'is_same_color_Антенны и аксессуары',
	'is_same_color_Аксессуары для игровых приставок',
	'is_same_color_Оперативная память',
	'is_same_color_Рюкзаки, чехлы, сумки',
	'is_same_color_Гаджет',
	'is_same_color_Магнитола',
	'is_same_color_MP3-плееры',
	'is_same_color_Видеокамеры',
	'is_same_color_Коврик для мыши',
	'is_same_color_Домофон',
	'is_same_color_Материнская плата',
	'is_same_color_Процессор',
	'is_same_color_Системы охлаждения для компьютеров',
	'is_same_color_ТВ-приставки и медиаплееры',
	'is_same_color_Штативы и головки',
	'is_same_color_Игровая приставка',
	'is_same_color_CD проигрыватели и плееры',
	'is_same_color_Запчасти и аксессуары для проекторов',
	'is_same_color_Блоки питания',
	'is_same_color_Сетевые адаптеры и PoE-инжекторы',
	'is_same_color_Виниловые проигрыватели и аксессуары',
	'is_same_color_Карты памяти и флешки',
	'is_same_color_Расходные материалы',
	'is_same_color_Запчасти для аудио/видеотехники',
	'is_same_color_Усилители, ресиверы и ЦАПы',
	'is_same_color_Микрофоны и аксессуары',
	'is_same_color_Охранная система',
	'is_same_color_Умный дом',
	'is_same_color_Часы и электронные будильники',
	'is_same_color_Графический планшет',
	'is_same_color_Навигаторы',
	'is_same_color_Очки виртуальной реальности',
	'is_same_color_Объектив',
	'is_same_color_Электронная книга',
	'is_same_color_Звуковая карта',
	'is_same_color_Домашний кинотеатр',
	'is_same_color_Проектор',
	'is_same_color_Сценическое оборудование и свет',
	'is_same_color_rest',
	'is_same_color_Аксессуары для квадрокоптеров',

    ## bert resnet
	'euclidean_bert_main_resnet_dist',
	'cosine_bert_main_resnet_resnet_dist',
	'jensenshannon_bert_main_resnet_resnet_dist',
	'minkowski_bert_main_resnet_resnet_dist',
	'sqeuclidean_bert_main_resnet_resnet_dist',

    ## minilm resnet
    "euclidean_minilm_main_resnet_dist",
	"cosine_minilm_main_resnet_resnet_dist",
	"jensenshannon_minilm_main_resnet_resnet_dist",
	"minkowski_minilm_main_resnet_resnet_dist",
	"sqeuclidean_minilm_main_resnet_resnet_dist",

    ## main pic resnet
	'euclidean_main_pic_embeddings_resnet_dist',
	'cosine_main_pic_embeddings_resnet_dist',
	'jensenshannon_main_pic_embeddings_resnet_dist',
	'minkowski_main_pic_embeddings_resnet_dist',
	'sqeuclidean_main_pic_embeddings_resnet_dist',

    ## dist
	'levenshtein_distance',
	'damerau_levenshtein_distance',
	'hamming_distance',
	'jaro_similarity',
	'jaro_winkler_similarity',
	'partial_ratio',
	'token_sort_ratio',
	'token_set_ratio',
	'w_ratio',
	'uq_ratio',
	'q_ratio',
	'matching_numbers',
	'matching_numbers_log',
	'log_fuzz_score'
]

categorical_feats = [
	'cat3_grouped1',
	'cat3_grouped2',
    'cat41',
	'cat42',
]

characteristic_feats = [
	'score_MP3-плееры_Объем встроенной памяти',
	'score_MP3-плееры_Управление воспроизведением',
	'score_SIM-карты_Домашний регион',
	'score_SIM-карты_Стартовый баланс, руб',
	'score_SIM-карты_Тип',
	'score_Автоматика телескопа',
	'score_Аксессуары для игровых приставок_Бренд',
	'score_Аксессуары для игровых приставок_Тип',
	'score_Аксессуары для квадрокоптеров_Название цвета',
	'score_Аксессуары для фото и видеотехники_Вес товара, г',
	'score_Аксессуары для фото и видеотехники_Винт под штативное гнездо',
	'score_Аксессуары для фото и видеотехники_Высота, см',
	'score_Аксессуары для фото и видеотехники_Размеры, мм',
	'score_Аксессуары для фото и видеотехники_Формат фона, м',
	'score_Акустика и колонки_Вес товара, г',
	'score_Акустика и колонки_Материал корпуса',
	'score_Акустика и колонки_Сопротивление, Ом',
	'score_Антенны и аксессуары_Коэффициент усиления антенны, dBi',
	'score_Артикул',
	'score_Батарейки и аккумуляторы_Вес товара, г',
	'score_Батарейки и аккумуляторы_Входные интерфейсы',
	'score_Батарейки и аккумуляторы_Гарантийный срок',
	'score_Батарейки и аккумуляторы_Емкость, мА•ч',
	'score_Батарейки и аккумуляторы_Кол-во циклов заряд-разряд',
	'score_Батарейки и аккумуляторы_Количество в упаковке, шт',
	'score_Батарейки и аккумуляторы_Назначение',
	'score_Батарейки и аккумуляторы_Размеры, мм',
	'score_Батарейки и аккумуляторы_Тип',
	'score_Батарейки и аккумуляторы_Цвет товара',
	'score_Бесконтактная оплата',
	'score_Беспроводные интерфейсы',
	'score_Бленда',
	'score_Бренд',
	'score_Вариант',
	'score_Вес товара, г',
	'score_Взаимодействие с устройствами',
	'score_Вид запчасти',
	'score_Вид стабилизатора',
	'score_Видеокамеры_Время непрерывной работы, мин',
	'score_Видеокамеры_Длина кабеля, м',
	'score_Видеокамеры_Комплектация',
	'score_Видеокамеры_Скорость видеосъемки 4K, кадр/с',
	'score_Видеокарты и графические ускорители_Бренд',
	'score_Видеокарты и графические ускорители_Гарантийный срок',
	'score_Видеокарты и графические ускорители_Размеры, мм',
	'score_Видеокарты и графические ускорители_Ревизия',
	'score_Видеокарты и графические ускорители_Серия графического процессора',
	'score_Видеокарты и графические ускорители_Тип памяти',
	'score_Видеокарты и графические ускорители_Частота графического процессора, МГц',
	'score_Видеокарты и графические ускорители_Частота работы шейдерных блоков, МГц',
	'score_Видеокарты и графические ускорители_Частота шины памяти, МГц',
	'score_Видеокарты и графические ускорители_Число текстурных блоков',
	'score_Видеонаблюдение_Габариты камеры (ДхШхВ, мм)',
	'score_Видеонаблюдение_Суммарный объем всех дисков, ГБ',
	'score_Видеонаблюдение_Угол обзора, градусов',
	'score_Видеонаблюдение_Упаковка',
	'score_Видеонаблюдение_Цвет товара',
	'score_Видеонаблюдение_Цветность дисплея',
	'score_Видеорегистратор_Защищенность',
	'score_Видеорегистратор_Конструкция видеорегистратора',
	'score_Видеорегистратор_Контроль движения',
	'score_Видеорегистратор_Размеры, мм',
	'score_Виниловые проигрыватели и аксессуары_Страна-изготовитель',
	'score_Внешняя оболочка кабеля',
	'score_Время зарядки аккумулятора, мин',
	'score_Время непрерывной работы, мин',
	'score_Время работы в режиме разговора, ч',
	'score_Время работы от аккумулятора, дни',
	'score_Гаджет_Бренд',
	'score_Гаджет_Цвет товара',
	'score_Гарантийный срок',
	'score_Глубина, мм',
	'score_Глубина, см',
	'score_Диаметр, мм',
	'score_Динамическая контрастность',
	'score_Длина ремешка, мм',
	'score_Длина, м',
	'score_Длина, см',
	'score_Домашний регион',
	'score_Домофон_Вес товара, г',
	'score_Домофон_Камера',
	'score_Домофон_Комплектация',
	'score_Домофон_Разрешение камеры',
	'score_Доп. комплектация',
	'score_Дополнительная информация по уцененному товару',
	'score_Дополнительные свойства покрытия',
	'score_Единиц в одном товаре',
	'score_Емкость аккумулятора, мАч',
	'score_Емкость, мА•ч',
	'score_Жесткие диски, SSD и сетевые накопители_Скорость вращения шпинделя HDD',
	'score_Жесткие диски, SSD и сетевые накопители_Среднее время задержки (Latency), мс',
	'score_Жесткие диски, SSD и сетевые накопители_Тип',
	'score_Запчасти для аудио/видеотехники_Бренд',
	'score_Запчасти для аудио/видеотехники_Комплектация',
	'score_Запчасти для аудио/видеотехники_Цвет товара',
	'score_Запчасти для ноутбуков_Диагональ экрана, дюймы',
	'score_Запчасти для ноутбуков_Длина кабеля, см',
	'score_Запчасти для ноутбуков_Материал',
	'score_Запчасти для ноутбуков_Партномер',
	'score_Запчасти для ноутбуков_Совместимость',
	'score_Запчасти для ноутбуков_Технология матрицы',
	'score_Запчасти для ноутбуков_Тип',
	'score_Запчасти для смартфонов_Бренд',
	'score_Запчасти для смартфонов_Вес товара, г',
	'score_Запчасти для смартфонов_Тип',
	'score_Запчасти для смартфонов_Цвет товара',
	'score_Запчасти и аксессуары для проекторов_Диагональ экрана, см',
	'score_Запчасти и аксессуары для проекторов_Комплектация',
	'score_Зарядные устройства и док-станции_Бренд',
	'score_Зарядные устройства и док-станции_Вес товара, г',
	'score_Зарядные устройства и док-станции_Гарантийный срок',
	'score_Зарядные устройства и док-станции_Интерфейсы',
	'score_Зарядные устройства и док-станции_Макс. выходная мощность, Вт',
	'score_Зарядные устройства и док-станции_Назначение',
	'score_Зарядные устройства и док-станции_Тип',
	'score_Зарядные устройства и док-станции_Цвет товара',
	'score_Защитные пленки и стекла_Бренд',
	'score_Защитные пленки и стекла_Гарантийный срок',
	'score_Защитные пленки и стекла_Комплектация',
	'score_Защитные пленки и стекла_Назначение',
	'score_Защитные пленки и стекла_Страна-изготовитель',
	'score_Защитные пленки и стекла_Толщина стекла, мм',
	'score_Звуковая карта_Усиление, мощность',
	'score_ИБП_Активная мощность, Вт',
	'score_ИБП_Высота, мм',
	'score_ИБП_Мощность, Вт',
	'score_ИБП_Ширина, мм',
	'score_Игровая приставка_Бренд',
	'score_Игровая приставка_Количество разъемов USB 3.1',
	'score_Игровая приставка_Страна-изготовитель',
	'score_Интерфейс подключения',
	'score_Интерфейсы',
	'score_Интерфейсы и разъемы',
	'score_Интерфейсы камеры',
	'score_Кабели и переходники_Вес товара, г',
	'score_Кабели и переходники_Внешняя оболочка кабеля',
	'score_Кабели и переходники_Длина, м',
	'score_Кабели и переходники_Материал проводника',
	'score_Кабели и переходники_Стандарт быстрой зарядки',
	'score_Кабели и переходники_Тип',
	'score_Кабели и переходники_Цвет товара',
	'score_Кабели и переходники_Электробезопасность',
	'score_Карты памяти и флешки_Бренд',
	'score_Карты памяти и флешки_Класс скорости',
	'score_Карты памяти и флешки_Макс. Скорость записи, Мб/с',
	'score_Карты памяти и флешки_Тип',
	'score_Карты памяти и флешки_Тип карты памяти',
	'score_Класс скорости',
	'score_Коврик для мыши_Вес товара, г',
	'score_Коврик для мыши_Длина, см',
	'score_Коврик для мыши_Название цвета',
	'score_Коврик для мыши_Страна-изготовитель',
	'score_Коврик для мыши_Ширина, см',
	'score_Кол-во циклов заряд-разряд',
	'score_Количество в упаковке, шт',
	'score_Количество вентиляторов',
	'score_Количество клавиш клавиатуры',
	'score_Количество копий, шт.',
	'score_Количество основных камер',
	'score_Количество портов',
	'score_Количество разъемов USB 2.0',
	'score_Количество слотов RAM',
	'score_Комплектация',
	'score_Комплектация процессора',
	'score_Компьютер_Видеопамять',
	'score_Компьютер_Гарантийный срок',
	'score_Компьютер_Картридер',
	'score_Компьютер_Конфигурация',
	'score_Компьютер_Мощность блока питания, Вт',
	'score_Компьютер_Назначение',
	'score_Компьютер_Общий объем HDD, ГБ',
	'score_Компьютер_Общий объем SSD, ГБ',
	'score_Компьютер_Оперативная память',
	'score_Компьютер_Размеры, мм',
	'score_Компьютер_Суммарный объем всех дисков, ГБ',
	'score_Компьютер_Частота процессора, ГГц',
	'score_Компьютер_Число портов eSATA',
	'score_Контроль движения',
	'score_Конфигурация',
	'score_Корпуса для компьютеров_Количество разъемов USB 2.0',
	'score_Кронштейн_Комплектация',
	'score_Кронштейн_Размеры, мм',
	'score_Макс. время записи, часов',
	'score_Макс. время работы (видео),  ч',
	'score_Макс. выходная мощность, Вт',
	'score_Макс. диагональ, дюймы',
	'score_Макс. нагрузка, Вт',
	'score_Макс. объем карты памяти,  ГБ',
	'score_Макс. поддерживаемое разрешение',
	'score_Макс. разрешение видеозаписи',
	'score_Макс. разрешение датчика, dpi',
	'score_Макс. разрешение копира (ч/б)',
	'score_Макс. скорость видеосъемки, кадр/с',
	'score_Макс. увеличение, крат',
	'score_Максимальное время работы, ч',
	'score_Материал браслета/ремешка',
	'score_Материал проводника',
	'score_Материнская плата_Бренд',
	'score_Материнская плата_Гарантийный срок',
	'score_Материнская плата_Количество PCI-E x4',
	'score_Материнская плата_Макс. поддерживаемая частота RAM',
	'score_Материнская плата_Размеры, мм',
	'score_Материнская плата_Тип',
	'score_Место крепления',
	'score_Микрофоны и аксессуары_Цвет товара',
	'score_Мин. входное напряжение, В',
	'score_Модель процессора',
	'score_Модель устройства',
	'score_Модуль связи WiFi',
	'score_Мониторы и запчасти_Динамическая контрастность',
	'score_Мониторы и запчасти_Яркость, кд/м2',
	'score_Мощность нагрузки, Вт',
	'score_Мощность, Вт',
	'score_Нагрузка в месяц, страниц',
	'score_Название модели',
	'score_Название цвета',
	'score_Назначение',
	'score_Назначение слотов',
	'score_Направляющая в комплекте',
	'score_Наушники и гарнитуры_Время зарядки аккумулятора, мин',
	'score_Наушники и гарнитуры_Комплектация',
	'score_Наушники и гарнитуры_Макс. частота, Гц',
	'score_Наушники и гарнитуры_Название модели',
	'score_Наушники и гарнитуры_Сопротивление, Ом',
	'score_Наушники и гарнитуры_Шумоподавление',
	'score_Образец цвета',
	'score_Общий объем SSD, ГБ',
	'score_Объектив_Бленда',
	'score_Объектив_Вес товара, г',
	'score_Объектив_Размеры, мм',
	'score_Объектив_Фокусное расстояние',
	'score_Объем',
	'score_Объем встроенной памяти',
	'score_Объем, мл',
	'score_Оперативная память',
	'score_Оперативная память_RAS to CAS Delay (tRCD)',
	'score_Оперативная память_Тайминги',
	'score_Оптические приборы_Бренд',
	'score_Оптические приборы_Гарантийный срок',
	'score_Оптические приборы_Комплектация',
	'score_Оптические приборы_Макс. увеличение, крат',
	'score_Оптические приборы_Страна-изготовитель',
	'score_Оптические приборы_Тип',
	'score_Оптические приборы_Фокусное расстояние, мм',
	'score_Офисная техника_Максимальная плотность бумаги, г/м2',
	'score_Офисная техника_Минимальная плотность бумаги, г/м2',
	'score_Очки виртуальной реальности_Образец цвета',
	'score_Партномер',
	'score_Печать фотографий',
	'score_Платформа',
	'score_Поддержка eSim',
	'score_Подключение к Smart TV',
	'score_Подключение к нескольким устройствам',
	'score_Полная мощность, В·А',
	'score_Потребляемая мощность, Вт',
	'score_Применение',
	'score_Принтеры и МФУ_Нагрузка в месяц, страниц',
	'score_Принтеры и МФУ_СНПЧ',
	'score_Проектор_Вес товара, г',
	'score_Проектор_Комплектация',
	'score_Процессор',
	'score_Процессор_Базовая частота, ГГц',
	'score_Процессор_Бренд',
	'score_Процессор_Кэш L3, МБ',
	'score_Пульты ДУ в комплекте',
	'score_Работа в режиме ожидания, ч',
	'score_Размеры, мм',
	'score_Разрешение',
	'score_Разрешение камеры',
	'score_Раскладка клавиатуры',
	'score_Расположение подсветки',
	'score_Расходник для печати_Бренд',
	'score_Расходник для печати_Количество в упаковке, шт',
	'score_Расходник для печати_Оригинальность расходника',
	'score_Расходник для печати_Ресурс',
	'score_Расходник для печати_Цвет товара',
	'score_Расходник для печати_Цвет тонера/чернил',
	'score_Расходные материалы_Диаметр, мм',
	'score_Расходные материалы_Длина, см',
	'score_Расходные материалы_Комплектация',
	'score_Расходные материалы_Объем, л',
	'score_Расходные материалы_Страна-изготовитель',
	'score_Расходные материалы_Твердость по Шору',
	'score_Расходные материалы_Технология 3D печати',
	'score_Расходные материалы_Цвет товара',
	'score_Режимы съемки',
	'score_Рекомендовано для',
	'score_Рюкзаки, чехлы, сумки_Внешние размеры, мм',
	'score_Рюкзаки, чехлы, сумки_Пол',
	'score_Рюкзаки, чехлы, сумки_Тип',
	'score_Сетевое оборудование_Емкость аккумулятора, мАч',
	'score_Сетевое оборудование_Максимальное время работы, ч',
	'score_Сетевое оборудование_Флеш-память, МБ',
	'score_Сетевые адаптеры и PoE-инжекторы_Бренд',
	'score_Сетевые адаптеры и PoE-инжекторы_Страна-изготовитель',
	'score_Сетевые адаптеры и PoE-инжекторы_Тип',
	'score_Сетевые возможности',
	'score_Сетевые фильтры, разветвители и удлинители_Входное напряжение, В',
	'score_Сетевые фильтры, разветвители и удлинители_Комплектация',
	'score_Сетевые фильтры, разветвители и удлинители_Макс. нагрузка, Вт',
	'score_Сетевые фильтры, разветвители и удлинители_Сечение жилы, кв.мм',
	'score_Сетевые фильтры, разветвители и удлинители_Стандарт защиты',
	'score_Системы охлаждения для компьютеров_Размеры, мм',
	'score_Скорость видеосъемки 4K, кадр/с',
	'score_Скорость открывания, м/мин',
	'score_Скорость сетевого адаптера',
	'score_Смарт-часы_Бренд',
	'score_Смарт-часы_Гарантийный срок',
	'score_Смарт-часы_Материал браслета/ремешка',
	'score_Смарт-часы_Модель браслета/умных часов',
	'score_Смарт-часы_Название цвета',
	'score_Смарт-часы_Образец цвета',
	'score_Смарт-часы_Оповещения',
	'score_Смарт-часы_Процессор, МГц',
	'score_Смарт-часы_Работа в режиме ожидания, ч',
	'score_Смарт-часы_Размер циферблата',
	'score_Смарт-часы_Размеры, мм',
	'score_Смарт-часы_Рекомендовано для',
	'score_Смарт-часы_Страна-изготовитель',
	'score_Смарт-часы_Форма циферблата',
	'score_Смарт-часы_Ширина ремешка, мм',
	'score_Смартфоны, планшеты, мобильные телефоны_Беспроводные интерфейсы',
	'score_Смартфоны, планшеты, мобильные телефоны_Бренд',
	'score_Смартфоны, планшеты, мобильные телефоны_Версия iOS',
	'score_Смартфоны, планшеты, мобильные телефоны_Вес товара, г',
	'score_Смартфоны, планшеты, мобильные телефоны_Встроенная память',
	'score_Смартфоны, планшеты, мобильные телефоны_Встроенные датчики',
	'score_Смартфоны, планшеты, мобильные телефоны_Емкость аккумулятора, мАч',
	'score_Смартфоны, планшеты, мобильные телефоны_Количество основных камер',
	'score_Смартфоны, планшеты, мобильные телефоны_Комплектация',
	'score_Смартфоны, планшеты, мобильные телефоны_Макс. объем карты памяти,  ГБ',
	'score_Смартфоны, планшеты, мобильные телефоны_Макс. скорость видеосъемки, '
	'кадр/с',
	'score_Смартфоны, планшеты, мобильные телефоны_Модель процессора',
	'score_Смартфоны, планшеты, мобильные телефоны_Модуль связи WiFi',
	'score_Смартфоны, планшеты, мобильные телефоны_Название цвета',
	'score_Смартфоны, планшеты, мобильные телефоны_Назначение слотов',
	'score_Смартфоны, планшеты, мобильные телефоны_Оперативная память',
	'score_Смартфоны, планшеты, мобильные телефоны_Особенности',
	'score_Смартфоны, планшеты, мобильные телефоны_Поддержка eSim',
	'score_Смартфоны, планшеты, мобильные телефоны_Процессор',
	'score_Смартфоны, планшеты, мобильные телефоны_Размеры, мм',
	'score_Смартфоны, планшеты, мобильные телефоны_Режимы съемки',
	'score_Смартфоны, планшеты, мобильные телефоны_Тип',
	'score_Смартфоны, планшеты, мобильные телефоны_Тип карты памяти',
	'score_Смартфоны, планшеты, мобильные телефоны_Цвет товара',
	'score_Совместимость',
	'score_Совместимые модели',
	'score_Совместимые пылесосы',
	'score_Сокет процессора',
	'score_Сопротивление, Ом',
	'score_Срок годности тарифа в днях',
	'score_Стабилизатор',
	'score_Стандарт защиты',
	'score_Стартовый баланс, руб',
	'score_Стекло',
	'score_Степень защиты',
	'score_Страна-изготовитель',
	'score_Суммарный объем всех дисков, ГБ',
	'score_Суммарный объем памяти',
	'score_Сценическое оборудование и свет_Количество режимов',
	'score_Сценическое оборудование и свет_Размеры, мм',
	'score_ТВ-приставки и медиаплееры_Тип',
	'score_Тайминги',
	'score_Твердость по Шору',
	'score_Телевизоры_Вес товара, г',
	'score_Телевизоры_Сетевые возможности',
	'score_Тепловыделение, Вт',
	'score_Тип витой пары',
	'score_Тип жесткого диска',
	'score_Тип карты памяти',
	'score_Тип памяти',
	'score_Тип привода ворот',
	'score_Тип соединения',
	'score_Умный дом_Пульты ДУ в комплекте',
	'score_Умный дом_Состав комплекта',
	'score_Умный дом_Тип',
	'score_Умный дом_Тип привода ворот',
	'score_Управление',
	'score_Усиление, мощность',
	'score_Устройство ручного ввода_Артикул',
	'score_Устройство ручного ввода_Вес товара, г',
	'score_Устройство ручного ввода_Комплектация',
	'score_Устройство ручного ввода_Макс. разрешение датчика, dpi',
	'score_Устройство ручного ввода_Питание',
	'score_Устройство ручного ввода_Подключение к Smart TV',
	'score_Устройство ручного ввода_Подключение к нескольким устройствам',
	'score_Устройство ручного ввода_Раскладка клавиатуры',
	'score_Устройство ручного ввода_Тип связи',
	'score_Устройство ручного ввода_Форм-фактор батареи',
	'score_Флеш-память, МБ',
	'score_Форм-фактор RAM',
	'score_Форма циферблата',
	'score_Формат фона, м',
	'score_Форматы аудио',
	'score_Форматы видео',
	'score_Фотоаппараты_Общее количество пикселей',
	'score_Фотоаппараты_Физический размер матрицы',
	'score_Функции',
	'score_Функциональные особенности оптического прибора',
	'score_Цвет ремешка',
	'score_Цвет товара',
	'score_Цвет тонера/чернил',
	'score_Частота графического процессора, МГц',
	'score_Частота процессора, ГГц',
	'score_Частота работы шейдерных блоков, МГц',
	'score_Частота шины памяти, МГц',
	'score_Часы и электронные будильники_Бренд',
	'score_Часы и электронные будильники_Тип',
	'score_Часы и электронные будильники_Цвет товара',
	'score_Чехол_Бренд',
	'score_Чехол_Вид чехла',
	'score_Чехол_Гарантийный срок',
	'score_Чехол_Комплектация',
	'score_Чехол_Макс. диагональ, дюймы',
	'score_Чехол_Модель браслета/умных часов',
	'score_Чехол_Модель устройства',
	'score_Чехол_Название цвета',
	'score_Чехол_Страна-изготовитель',
	'score_Чехол_Тип',
	'score_Чехол_Цвет товара',
	'score_Число портов eSATA',
	'score_Число текстурных блоков',
	'score_Ширина, мм',
	'score_Ширина, см',
	'score_Шумоподавление',
	'score_Электробезопасность',
	'score_Электронная книга_Бренд',
	'score_Электронная книга_Комплектация',
	'score_Электронная книга_Разрешение экрана',
	'score_Электронная книга_Форматы аудио',
	'score_Электронные модули_Бренд',
	'score_Электронные модули_Гарантийный срок',
	'score_Электронные модули_Страна-изготовитель',
	'score_Электронные модули_Тип',
	'score_Яркость, кд/м2'
]