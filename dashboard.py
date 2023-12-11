import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import numpy as np

st.set_page_config(layout='centered', page_title='Dashboard by Alexey Shevchenko', page_icon="📈")
st.set_option('deprecation.showPyplotGlobalUse', False)

def preparing_data_if_error_loading(filename):
    '''
    Дело в том, что я столкнулся с такой проблемой при загрузке данных ->
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xca in position 1: invalid continuation byte

    Это было исправлено в начале notebook. Если необходимость в таком имеется только у меня, необходимо 
    будет в функции load_data изменить загрузку только на:
        pd.read_csv(filename, sep=",")
    '''
    data = pd.read_csv(filename, sep = ",", encoding='windows-1251')
    data["sep"] = data["Количество больничных дней,\"Возраст\",\"Пол\""].str.split(",")
    col = []
    age = []
    sex = []

    for i in range(data.shape[0]):
        rowSep = data.iloc[i]["sep"]
        col.append(int(rowSep[0]))
        age.append(int(rowSep[1]))
        sex.append(rowSep[2][1])
        data["Количество больничных дней"] = pd.Series(col)
        data["Возраст"] =  pd.Series(age)
        data["Пол"] =  pd.Series(sex)

    data = data.drop(["Количество больничных дней,\"Возраст\",\"Пол\"", "sep"], axis=1)
    return data

# Download data
@st.cache_data
def load_data(filename):
    data = preparing_data_if_error_loading(filename) # Если ошибка имеется
    # data = data.drop("Unnamed: 0", axis=1) # Если ошибки нет
    return data

data_file = st.file_uploader("Upload CSV", type="csv")
if data_file:
    with st.sidebar:
        st.header("Ввод параметров")
        medical_days = st.text_input("Количество рабочих дней", "2")
        age = st.text_input("Возраст", "35")

    medical_days = int(medical_days)
    age = int(age)

    st.subheader("Выбранные значения:")
    st.write("Количество рабочих дней:", medical_days)
    st.write("Возраст:", age)

    data = load_data(data_file)
    data.rename(columns={'Количество больничных дней': 'medical_days', 'Возраст': 'age', 'Пол': 'sex'}, inplace=True)

    st.write("* Загруженные данные:")
    st.table(data.head())
    st.write("* Описание данных:")
    st.dataframe(data.describe())
    number_data = data.drop('sex', axis=1)
    st.write(f"Дисперсия по рабочим дням: {round(number_data.var()[0], 2)}")
    st.write(f"Дисперсия по возрасту: {round(number_data.var()[1], 2)}")
    st.write(f"Корреляция величин: {round(number_data.corr().iloc[0,1], 2)}")
    st.write("Уровень значимости: 0.05")

    st.header(f"Проверка 1 гипотезы: Мужчины пропускают в течение года более {medical_days} рабочих дней по болезни значимо чаще женщин.")

    N1, N2 = data.sex.value_counts()
    st.write(f"* Количество мужчин: {N1}")
    st.write(f"* Количество женщин: {N2}")

    bins = np.arange(0, 10, 1)
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='medical_days', hue='sex', kde=True, bins=bins)
    plt.title('Распределение пропущенных дней по полу')
    st.pyplot()

    sns.histplot(
        data[data.sex == 'М'].medical_days, bins=bins,
        color='blue', alpha=0.5, label='M', stat='probability', kde=True)
    sns.histplot(
        data[data.sex == 'Ж'].medical_days, bins=bins,
        color='orange', alpha=0.5, label='Ж', stat='probability', kde=True)
    plt.legend(loc=1)
    plt.title('Распределение вероятностей по полу')
    st.pyplot()

    t_statistic, p_value = ttest_ind(
        data[(data.sex == 'М') & (data.medical_days > medical_days)].medical_days,
        data[(data.sex == 'Ж') & (data.medical_days > medical_days)].medical_days,
        equal_var=False,
        alternative='less'
    )

    st.write(f'* Гипотеза 1: Мужчины пропускают в течение года более {medical_days} рабочих дней по болезни значимо чаще женщин.')
    st.write('p-value:', p_value)
    st.write('  statistic:', t_statistic)

    st.write("Отвергаем гипотезу, так как p_value меньше уровня значимости 0.05" if p_value >= 0.05 else "Принимаем гипотезу, так как p_value больше уровня значимости 0.05")

    st.header(f"Проверка 2 гипотезы: Работники старше {age} лет пропускают в течение года более {medical_days} рабочих дней (medical_days) по болезни значимо чаще своих более молодых коллег.")

    data["age_type"] = ['old' if x > age else 'young' for x in data['age']]

    N3, N4 = data.age_type.value_counts()
    st.write(f"* Количество сотрудников старше {age} лет: {N3}")
    st.write(f"* количество сотрудников младше {age} лет: {N4}")

    sns.boxplot(x='age', y='medical_days', data=data)
    plt.title('Распределение пропущенных дней по возрасту')
    st.pyplot()

    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, hue='age_type', x='medical_days', kde=True)
    plt.title('Распределение пропущенных дней по возрасту категориально')
    st.pyplot()

    sns.histplot(
        data[data.age_type == 'old'].medical_days, bins=bins,
        color='blue', alpha=0.5, label='Old', stat='probability', kde=True)
    sns.histplot(
        data[data.age_type == 'young'].medical_days, bins=bins,
        color='orange', alpha=0.5, label='Young', stat='probability', kde=True)
    plt.legend(loc=1)
    plt.title('Распределение вероятностей по возрасту категориально')
    st.pyplot()

    t_statistic, p_value = ttest_ind(
        data[(data.age_type == 'old') & (data.medical_days > medical_days)].medical_days,
        data[(data.age_type == 'young') & (data.medical_days > medical_days)].medical_days,
        equal_var=False,
        alternative='less'
    )

    st.write(f'Гипотеза 2: Работники старше {age} лет пропускают в течение года более {medical_days} рабочих дней по болезни значимо чаще своих более молодых коллег.')
    st.write('p-value:', p_value)
    st.write('statistic:', t_statistic)

    st.write("Отвергаем гипотезу, так как p_value меньше уровня значимости 0.05" if p_value >= 0.05 else "Принимаем гипотезу, так как p_value больше уровня значимости 0.05")


