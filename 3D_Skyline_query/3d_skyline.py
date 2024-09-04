import numpy as np
import matplotlib.pyplot as plt
from skyline.skyline import dnc_skyline
from skyline.plotting import plot_skyline_with_marker  # 시각화 기능을 skyline 모듈에서 가져옴
from src.data_generation import generate_random_data  # 데이터 생성 함수


# 레이어를 할당하는 함수
def assign_layer(point):
    if point[0] > 125 or point[1] > 199 or point[2] > 6.4:
        return 3  # Diabetes
    elif point[0] > 99 or point[1] > 140 or point[2] > 5.7:
        return 2  # Prediabetes
    else:
        return 1  # Normal


def setdiff2d(array1, array2):
    flat_array2 = array2.flatten()

    # 중복되지 않는 요소를 찾기 위한 마스크 생성
    mask = np.isin(array1, flat_array2, invert=True)

    # 중복되지 않는 요소만 남기고 배열 형태 유지
    result = array1[mask]

    # 결과를 원본 형태로 다시 재구성
    if result.size == 0:
        return np.empty((0, array1.shape[1]))  # 비어있는 배열 반환

    return result.reshape(-1, array1.shape[1])
def main():
    num_samples = 50  # 랜덤 데이터 개수 설정
    data = generate_random_data(num_samples=num_samples)  # 데이터 생성

    layers = np.array([assign_layer(point) for point in data])

    layer1 = data[layers == 1]  # Normal layer
    layer2 = data[layers == 2]  # Prediabetes layer
    layer3 = data[layers == 3]  # Diabetes layer

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # 각 레이어에 대해 DNC 알고리즘을 사용하여 스카이라인을 계산하고, 좌표 표시
    if len(layer1)>0:
        skyline1 = dnc_skyline(layer1)

        other_point=setdiff2d(layer1,skyline1)
        plot_skyline_with_marker(skyline1, ax, 'green', 's', 'Normal_Skyline')
        plot_skyline_with_marker(other_point, ax, 'lightgreen', 's', 'Normal_Point',draw_lines=False)



    if len(layer2)>0:
        skyline2 = dnc_skyline(layer2)
        other_point = setdiff2d(layer2, skyline2)
        plot_skyline_with_marker(skyline2, ax, 'blue', 'o', 'Pre Diabetes_Skyline')
        plot_skyline_with_marker(other_point, ax, 'skyblue', 'o', 'Pre Diabetes',draw_lines=False)



    if len(layer3)>0:
        skyline3 = dnc_skyline(layer3)
        other_point = setdiff2d(layer3, skyline3)
        plot_skyline_with_marker(skyline3, ax, 'red', '^', 'Diabetes_Skyline points',draw_lines=False)
        plot_skyline_with_marker(other_point, ax, 'pink', '^', 'Diabetes',draw_lines=False)


    ax.set_xlabel('Fasting Glucose (mg/dL)')
    ax.set_ylabel('Postprandial Glucose (mg/dL)')
    ax.set_zlabel('HbA1c (%)')
    ax.set_xlim(70, 160)
    ax.set_ylim(0, 300)
    ax.set_zlim(0, 9.5)
    ax.set_title('Layered Skyline 3D Visualization using DNC Algorithm')

    ax.text2D(0.05, 0.95, f"Data: {num_samples}", transform=ax.transAxes, fontsize=12, color='black')

    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.show()

if __name__ == "__main__":
    main()
