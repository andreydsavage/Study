Запускаем приложение фаст апи локально из папки api: uvicorn app.main:app --host 0.0.0.0 --port 3400

Запускаем стримлит: streamlit run front/main.py


front содержит файл со стримлитом: кнопка загрузки и классифицирования изображения

api/app описание приложения

api/utils: 
- imagenet-simple-labels.json наименование классов
- resnet18-weights.pth веса модели скачанные с сайта
- model_func.py функции модели: нормализация, загрузка наименований классов, весов модели и трансформация изображения
преодобученная модель resnet18
