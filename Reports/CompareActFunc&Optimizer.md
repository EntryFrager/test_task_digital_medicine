# Анализ влияния активаций, нормализации и оптимизаторов на обучение простой сети для MNIST

В этом проекте реализована простая полносвязная нейронная сеть на NumPy и проведён комплексный анализ:

- **Активационные функции:** ReLU, LeakyReLU, ELU, SoftPlus, LogSoftMax  
- **Batch Normalization:** Batch Normalization + ChannelwiseScaling  
- **Оптимизаторы:** SGD с momentum и Adam  
- **Данные:** MNIST (28×28 → 100 → 10)

---

## Результаты

| Модель                        | Точность на тесте (%) | Замечания                                                   |
|-------------------------------|-----------------------|-------------------------------------------------------------|
| ReLU                          | 97.52                 | Быстрое и стабильное обучение                               |
| LeakyReLU                     | 97.40                 | Аналогично ReLU, чуть более мягкое поведение                |
| ELU                           | 97.40                 | Небольшие флуктуации в середине обучения                   |
| SoftPlus                      | 96.58                 | Более «мягкая» функция, медленнее сходимость                |
| ReLU + BN                     | 96.62                 | Быстрый старт, но финальная точность не улучшилась          |
| LeakyReLU + BN                | 96.99                 | Лучший баланс скорости и стабильности с BN                  |
| ELU + BN                      | 96.93                 | Гладкость кривой, но чуть ниже точности                     |
| SoftPlus + BN                 | 96.14                 | Ускорение начального обучения, но финальная точность ниже   |
| ReLU + BN + Adam              | 97.37                 | Быстрое падение потерь, высокая стабильность                |
| ReLU + BN + SGD with momentum | 97.32                 | Плавная сходимость, чуть ниже скорость старта, хорошая финальная точность |

## Влияние Batch Normalization на скорость обучения

При добавлении `BN` обучение в первые несколько эпох идёт заметно быстрее: уже к 1–2-й эпохе валидационная потеря оказывается существенно ниже, чем без `BN`. Это особенно заметно для `LeakyReLU` и `ELU`: без `BN` они стартовали с потерь ~0.25–0.30, а с `BN` — ~0.15–0.17.

## Улучшает ли BN точность на тесте?
В наших экспериментах `BN` не дал заметного прироста финальной тестовой точности. Наилучшие комбинации с `BN` всё же дают близкие к `без‑BN` результаты: `LeakyReLU+BN` ≈97.0%, `ReLU+BN` ≈96.6%.

## Влияние BN на стабильность обучения
С `BN` валидационная потеря колеблется меньше: исчезают крупные «провалы» на 5–7 эпохах, обучение идёт ровнее. Особенно это проявляется у `ELU` и `SoftPlus`: без `BN` они иногда «прыгают» вглубь валидационной кривой, а с `BN` — более гладко.

## Сходимость SGD с импульсом vs Adam
`Adam` в первых 3–5 эпохах снижает потерю быстрее, чем `SGD+momentum`: уже к 2–й эпохе `loss` ≈0.12 у `Adam` против ≈0.17 у `SGD`. 

`SGD` с импульсом «догоняет» к концу обучения и часто даёт чуть лучшую финальную точность (особенно при правильном подборе `learning rate` и `momentum`).

## Стабильность обучения: SGD vs Adam

`Adam` ведёт себя очень стабильно: loss-кривая почти без шумов, везде монотонно убывает.

`SGD+momentum` может демонстрировать небольшие колебания потерь в середине обучения (эпохи 5–10), особенно если `LR` слишком велик.

## Поведение функции потерь по эпохам
* **ReLU / LeakyReLU + SGD**: плавный логарифмический спад потерь, без резких скачков.

* **ELU / SoftPlus + SGD**: чуть более «пилообразный» спад, особенно на валидации.

* **Любая активация + BN**: очень резкий спад в 1–2 эпохи, затем более пологая фаза «тонкой настройки».

* **Adam (на ReLU+BN)**: потеря быстро падает в 1–3 эпохах и затем выравнивается, почти без шума.

## Основные выводы

1. **ReLU** показала лучшие результаты для простых полносвязных сетей на MNIST.  
2. **Batch Normalization**  
   - Ускоряет начальное обучение (резкий спад потерь в первые эпохи).  
   - Повышает устойчивость (меньше флуктуаций).
3. **LeakyReLU + BN** показала лучший компромисс между скоростью, стабильностью и точностью при использовании BN.  
4. **Adam**  
   - Быстро сходится в первые 3–5 эпох (минимальная настройка).  
   - Очень гладкие кривые потерь.  
5. **SGD + momentum**  
   - Медленнее старт, но при длительном обучении часто даёт чуть более высокую финальную точность.  
   - Небольшие колебания потерь в середине обучения при высоком LR.

