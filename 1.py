from typing import Tuple, List, Callable
import time
import numpy as np


def solve_task(x: np.ndarray, y: np.ndarray) -> Tuple[List[float], float]:
    """
    Находит все оптимальные пороговые значения для бинарного классификатора,
    минимизирующие среднеквадратичную ошибку (MSE).

    Параметры:
    -----------
    x : np.ndarray
        Массив числовых признаков формы (n,), где n - количество образцов.
        Каждое значение представляет собой точку данных для классификации.
    y : np.ndarray
        Массив бинарных меток формы (n,), где y_i ∈ {0, 1}.
        Метки должны соответствовать точкам x.

    Возвращает:
    -----------
    Tuple[List[float], float]
        Кортеж, содержащий:
        - Список всех оптимальных пороговых значений t
        - Минимальное значение MSE, соответствующее этим порогам

    Исключения:
    -----------
    ValueError
        - Если входные массивы имеют разную длину
        - Если массив x пустой
        - Если y содержит значения не из {0, 1}
    """
    if len(x) != len(y):
        raise ValueError("Длины массивов x и y должны совпадать")
    if len(x) == 0:
        raise ValueError("Входные массивы не могут быть пустыми")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Массив y должен содержать только 0 и 1")

    min_mse = {}
   
    all_t = list(set(x))
    all_t = np.append(all_t, -np.inf)
    all_t = np.append(all_t, np.inf)

    for t in all_t:
        mse = 0
        for i in range(len(x)):
            f = 1 if x[i] >= t else 0
            mse += (y[i] - f) ** 2
            
        mse /= len(x)
        min_mse.setdefault(mse, [])
        min_mse[mse].append(t)

    mse = min(min_mse)
    list_t = min_mse[mse]

    return list_t, mse


def validate_solution(x: np.ndarray, y: np.ndarray, solution_func: Callable) -> bool:
    """
    Проверяет, что решение удовлетворяет всем критериям оптимальности:
    1. MSE действительно минимально возможное
    2. Все возвращенные пороги дают одинаковое (минимальное) MSE
    """
    try:
        thresholds, mse = solution_func(x, y)

        # Проверка 1: MSE действительно минимальное
        all_possible_thresholds = [-np.inf] + list((np.sort(x)[:-1] + np.sort(x)[1:]) / 2) + [np.inf]
        possible_mses = []
        for t in all_possible_thresholds:
            pred = (x >= t).astype(int)
            possible_mses.append(np.mean((pred - y)**2))
        min_possible_mse = min(possible_mses)

        if not np.isclose(mse, min_possible_mse):
            return False

        # Проверка 2: Все пороги дают одинаковое MSE
        for t in thresholds:
            pred = (x >= t).astype(int)
            current_mse = np.mean((pred - y)**2)
            if not np.isclose(current_mse, mse):
                return False

        return True

    except Exception:
        raise

# Тестовые случаи
def test_standard_case_1(solution_func):
    x = np.array([1, 2, 3])
    y = np.array([0, 1, 0])
    assert validate_solution(x, y, solution_func)

def test_standard_case_2(solution_func):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 1, 0, 1, 1])
    assert validate_solution(x, y, solution_func)

def test_all_zeros(solution_func):
    x = np.array([5, 3, 4])
    y = np.array([0, 0, 0])
    assert validate_solution(x, y, solution_func)

def test_all_ones(solution_func):
    x = np.array([2, 5, 3])
    y = np.array([1, 1, 1])
    assert validate_solution(x, y, solution_func)

def test_duplicate_x(solution_func):
    x = np.array([2, 2, 2])
    y = np.array([0, 1, 0])
    assert validate_solution(x, y, solution_func)

def test_multiple_optimal_thresholds(solution_func):
    x = np.array([1, 2, 3, 4])
    y = np.array([0, 1, 1, 0])
    assert validate_solution(x, y, solution_func)

def test_performance(solution_func):
    n = 10**5
    np.random.seed(42)
    x = np.random.rand(n) * 100
    y = np.random.randint(0, 2, n)
    start = time.time()
    solution_func(x, y)
    duration = time.time() - start
    assert duration < 5

def test_single_element(solution_func):
    x = np.array([5])
    y = np.array([1])
    assert validate_solution(x, y, solution_func)

def test_all_negative_x(solution_func):
    x = np.array([-5, -3, -1])
    y = np.array([0, 1, 0])
    assert validate_solution(x, y, solution_func)

def test_mixed_positive_negative(solution_func):
    x = np.array([-2, 1, -3, 4])
    y = np.array([1, 0, 0, 1])
    assert validate_solution(x, y, solution_func)

def test_perfect_classifier(solution_func):
    x = np.array([1, 2, 3, 4])
    y = np.array([0, 0, 1, 1])
    assert validate_solution(x, y, solution_func)

def test_worst_classifier(solution_func):
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 1, 0, 0])
    assert validate_solution(x, y, solution_func)

def test_tied_thresholds(solution_func):
    x = np.array([1, 3, 5, 7])
    y = np.array([0, 1, 0, 1])
    assert validate_solution(x, y, solution_func)

def test_duplicate_x_mixed_labels(solution_func):
    x = np.array([2, 2, 2, 2])
    y = np.array([0, 0, 1, 1])
    assert validate_solution(x, y, solution_func)

def test_single_threshold_between_duplicates(solution_func):
    x = np.array([1, 1, 2, 2])
    y = np.array([0, 1, 1, 0])
    assert validate_solution(x, y, solution_func)

def test_floating_point_precision(solution_func):
    x = np.array([1.1, 2.2, 3.3, 4.4])
    y = np.array([0, 1, 0, 1])
    assert validate_solution(x, y, solution_func)

def test_large_value_range(solution_func):
    x = np.array([-1e6, 0, 1e6])
    y = np.array([0, 1, 1])
    assert validate_solution(x, y, solution_func)

def test_even_number_of_samples(solution_func):
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([0, 0, 1, 1, 0, 0])
    assert validate_solution(x, y, solution_func)

def test_odd_number_of_samples(solution_func):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 0, 1, 0, 0])
    assert validate_solution(x, y, solution_func)

def test_extreme_imbalance_labels(solution_func):
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert validate_solution(x, y, solution_func)

def test_multiple_identical_thresholds(solution_func):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 1, 0, 1, 0])
    assert validate_solution(x, y, solution_func)

def test_single_unique_x_multiple_labels(solution_func):
    x = np.array([5, 5, 5, 5])
    y = np.array([0, 1, 0, 1])
    assert validate_solution(x, y, solution_func)

def test_threshold_at_midpoint(solution_func):
    x = np.array([1, 3])
    y = np.array([0, 1])
    assert validate_solution(x, y, solution_func)

def test_no_optimal_threshold_in_data(solution_func):
    x = np.array([1.5, 2.5, 3.5])
    y = np.array([0, 1, 0])
    assert validate_solution(x, y, solution_func)

def test_all_labels_same_but_x_different(solution_func):
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 1, 1, 1])
    assert validate_solution(x, y, solution_func)

def test_high_dimensional_noise(solution_func):
    np.random.seed(42)
    x = np.random.randn(1000)
    y = (x > 0).astype(int)
    assert validate_solution(x, y, solution_func)

def test_non_integer_thresholds(solution_func):
    x = np.array([1.5, 2.5, 3.5, 4.5])
    y = np.array([0, 1, 0, 1])
    assert validate_solution(x, y, solution_func)

def test_edge_case_large_dataset(solution_func):
    x = np.concatenate([np.zeros(500), np.ones(500)])
    y = np.concatenate([np.zeros(500), np.ones(500)])
    assert validate_solution(x, y, solution_func)

def run_tests(solution_func: Callable):
    tests = [
        test_standard_case_1,
        test_standard_case_2,
        test_all_zeros,
        test_all_ones,
        test_duplicate_x,
        test_multiple_optimal_thresholds,
        test_performance,
        test_single_element,
        test_all_negative_x,
        test_mixed_positive_negative,
        test_perfect_classifier,
        test_worst_classifier,
        test_tied_thresholds,
        test_duplicate_x_mixed_labels,
        test_single_threshold_between_duplicates,
        test_floating_point_precision,
        test_large_value_range,
        test_even_number_of_samples,
        test_odd_number_of_samples,
        test_extreme_imbalance_labels,
        test_multiple_identical_thresholds,
        test_single_unique_x_multiple_labels,
        test_threshold_at_midpoint,
        test_no_optimal_threshold_in_data,
        test_all_labels_same_but_x_different,
        test_high_dimensional_noise,
        test_non_integer_thresholds,
        test_edge_case_large_dataset
    ]

    for test in tests:
        try:
            test(solution_func)
        except AssertionError as e:
            print(f"Ошибка в тесте {test.__name__}: {e}\n")
        except Exception as e:
            print(f"Непредвиденная ошибка в тесте {test.__name__}:\n{str(e)}\n")

    print('Все тесты завершены!')

run_tests(solve_task)