# Dog breeds recognizer

## Содержание

- [Вступление](#introduction)
- [Входные и выходные данные](#inout_data)
- [Алгоритм](#alg)
- [Подготовка данных и обучение](#dataset)
- [Результаты](#results)
- [Библиография](#biblio)

## <a id="introduction">Вступление</a>
Данная работа посвящена исследованию сверточной нейронной сети AlexNet[[1]](#bib1) для задачи распознования пород собак.
Результирующий продукт способен находить и распознавать собаку некоторой породу на изображении.
В рамках исследования была изучена результативность обучения сверточной нейронной сети на различных наборах данных.

## <a id="inout_data">Входные и выходные данные</a>
На вход готовому решению подаётся фотография в формате jpg, png или bmp. Желательно (для достижения большей точности) цветная и разрешение фрагмента содержащего собаку должно быть не менее 227*227.
Выходом программы является строка, являющаяся названием одной из пород собак.
Список пород:
- Afghan
- Airedale
- American Bullbog
- Basset
- Beagle
- Bermaise
- Bichon Frise
- Blenheim
- Bloodhound
- Bluetick
- Border Collie
- Borzoi
- Boxer
- Bull Mastiff
- Bull Terrier
- Bulldog
- Cairn
- Chihuahua
- Chinese Crested
- Chow
- Clumber
- Cockapoo
- Cocker
- Collie
- Corgi
- Dachshund
- Dalmation
- Doberman
- French Bulldog
- German Sheperd
- Golden Retriever
- Great Dane
- Greyhound
- Groenendael
- Irish Spaniel
- Irish Wolfhound
- Jack Russell Terrier
- Japanese Spaniel
- Kelpie
- Komondor
- Labrador
- Lhasa
- Malinois
- Maltese
- Pekinese
- Pit Bull
- Pomeranian
- Poodle
- Pug
- Rhodesian
- Rottweiler
- Saint Bernard
- Schnauzer
- Scotch Terrier
- Shar_Pei
- Shiba Inu
- Shih-Tzu
- Siberian Husky
- Staffordshire Bull Terrier
- Yorkie

## <a id="alg">Алгоритм</a>
1. Изображение загружается в помощью OpenCV и подается на вход сети yolo3[[2]](#bib2).
2. Вырезаются фрагменты изображения, на которых с уверенностью не менее 0.9 определяется наличие собаки.
3. Для каждого такого фрагмента запускается обученная сеть AlexNet, которая предполагает породу собаки на фрагменте.
4. Наивероятнейший ответ AlexNet для каждого фрагмента выводится на изображение с помощью OpenCV.
   
## <a id="dataset">Подготовка данных и обучение</a>
Для обучения сверточной нейронной сети было выбрано два набора данных, размешенных на платформе Kaggle: [первый](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set) и [второй](https://www.kaggle.com/datasets/enashed/dog-breed-photos).
Особенности первого набора:
- 9300 изображений
- 70 пород собак
- "чистые" данные
  - на каждом изображении ровно одна собака
  - собака занимает всё изображение
  - собака является типичным представителем породы
  - фотография сделана с информативного ракурса
  - все изображения одного разрешения (227*227)
- разметка данных - в наборе три директории (train, test, valid). В каждой из них расположены соответственно названные директории для каждой породы. В них уже содержатся изображения.
- равномерность данных - в тренировочном и валидационном наборах содержится ровно по 10 изображений представителей каждой породы. В тренировочном от 80 до 150 изображений для каждой породы.
Особенности второго набора:
- 117000 изображений
- более 300 уникальных строк для пород собак
- "грязные" данные
    - не на всех изображениях одна собака
    - собака занимает только некоторую часть изображения
    - достаточно большую долю занимают гибриды, помеси и беспородные собаки
    - некоторые собаки определены ошибочно
    - некоторые позы и ракурсы не позволяют определить породу собаки
- разметка данных - все изображения лежат вместе, соответствие имени файла и породы описано в общем csv файле
- неравномерность данных - от 1 до 2200 изображений для породы 

classifyed https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set
big https://www.kaggle.com/datasets/enashed/dog-breed-photos


## <a id="biblio">Библиография</a>
1. <a id="bib1"> Krizhevsky A., Sutskever I., Hinton G. E. Imagenet classification with deep convolutional neural networks //Advances in neural information processing systems. – 2012. – Т. 25.</a>
2. <a id="bib2"> Redmon J., Farhadi A. Yolov3: An incremental improvement //arXiv preprint arXiv:1804.02767. – 2018.</a>
