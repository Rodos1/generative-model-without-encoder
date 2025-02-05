# generative-model-without-encoder
Это что-то вроде бэклога по работе над курсовой. Тема: optical generative model

Пока что лучшее, что получалось (цифра 5):

![final_generated_image](https://github.com/user-attachments/assets/533240ca-c051-4f3a-8ddb-b81c0c25fa62)

Обучение происходило классическим backprop, никаких гиперпараметров вроде wavelength или distance не крутилось, изменялись только характеристики модуляторов. В пропагаторе использовались преобразования Фурье и дифракционный фильтр Френеля.
Прокрутка в Optuna гиперпараметров не дала никаких результатов


# Гиперпараметры
batch_size = 64
input_size = (28, 28)
num_layers = 2
wavelength = 400E-9
distance = 0.1
num_epochs = 200  # Максимальное количество эпох
lr = 0.001
loss_threshold = 0.02  # Значение Loss, при котором остановится обучение

Оптимизатор: Адам 
Функция ошибки: BCEWithLogitsLoss
