{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "**Task 2** Milestone 2"
      ],
      "metadata": {
        "id": "R4plqi9O0iU4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import pickle\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n",
        "import seaborn as sns \n",
        "import sklearn\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.model_selection import RandomizedSearchCV\n",
        "import imblearn\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score"
      ],
      "metadata": {
        "id": "46q0z5b-R-18"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#importing the dataset which is in csv file\n",
        "data=pd.read_csv('/content/Bank_Personal_Loan.csv')\n",
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 287
        },
        "id": "YExq5fAZO2U5",
        "outputId": "2eb0f10a-71ff-42eb-cf8f-46c6a462166d"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   ID  Age  Experience  Income  ZIP Code  Family  CCAvg  Education  Mortgage  \\\n",
              "0   1   25           1      49     91107       4    1.6          1         0   \n",
              "1   2   45          19      34     90089       3    1.5          1         0   \n",
              "2   3   39          15      11     94720       1    1.0          1         0   \n",
              "3   4   35           9     100     94112       1    2.7          2         0   \n",
              "4   5   35           8      45     91330       4    1.0          2         0   \n",
              "\n",
              "   Personal Loan  Securities Account  CD Account  Online  CreditCard  \n",
              "0              0                   1           0       0           0  \n",
              "1              0                   1           0       0           0  \n",
              "2              0                   0           0       0           0  \n",
              "3              0                   0           0       0           0  \n",
              "4              0                   0           0       0           1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-53218659-0a5d-4b8a-857d-6c9f4b435f44\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>ID</th>\n",
              "      <th>Age</th>\n",
              "      <th>Experience</th>\n",
              "      <th>Income</th>\n",
              "      <th>ZIP Code</th>\n",
              "      <th>Family</th>\n",
              "      <th>CCAvg</th>\n",
              "      <th>Education</th>\n",
              "      <th>Mortgage</th>\n",
              "      <th>Personal Loan</th>\n",
              "      <th>Securities Account</th>\n",
              "      <th>CD Account</th>\n",
              "      <th>Online</th>\n",
              "      <th>CreditCard</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>25</td>\n",
              "      <td>1</td>\n",
              "      <td>49</td>\n",
              "      <td>91107</td>\n",
              "      <td>4</td>\n",
              "      <td>1.6</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>45</td>\n",
              "      <td>19</td>\n",
              "      <td>34</td>\n",
              "      <td>90089</td>\n",
              "      <td>3</td>\n",
              "      <td>1.5</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>39</td>\n",
              "      <td>15</td>\n",
              "      <td>11</td>\n",
              "      <td>94720</td>\n",
              "      <td>1</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>35</td>\n",
              "      <td>9</td>\n",
              "      <td>100</td>\n",
              "      <td>94112</td>\n",
              "      <td>1</td>\n",
              "      <td>2.7</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>35</td>\n",
              "      <td>8</td>\n",
              "      <td>45</td>\n",
              "      <td>91330</td>\n",
              "      <td>4</td>\n",
              "      <td>1.0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-53218659-0a5d-4b8a-857d-6c9f4b435f44')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-53218659-0a5d-4b8a-857d-6c9f4b435f44 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-53218659-0a5d-4b8a-857d-6c9f4b435f44');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5dc1TIVIzI33",
        "outputId": "84fa4274-f944-4a1d-afca-d0bafe3dabdb"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 5000 entries, 0 to 4999\n",
            "Data columns (total 14 columns):\n",
            " #   Column              Non-Null Count  Dtype  \n",
            "---  ------              --------------  -----  \n",
            " 0   ID                  5000 non-null   int64  \n",
            " 1   Age                 5000 non-null   int64  \n",
            " 2   Experience          5000 non-null   int64  \n",
            " 3   Income              5000 non-null   int64  \n",
            " 4   ZIP Code            5000 non-null   int64  \n",
            " 5   Family              5000 non-null   int64  \n",
            " 6   CCAvg               5000 non-null   float64\n",
            " 7   Education           5000 non-null   int64  \n",
            " 8   Mortgage            5000 non-null   int64  \n",
            " 9   Personal Loan       5000 non-null   int64  \n",
            " 10  Securities Account  5000 non-null   int64  \n",
            " 11  CD Account          5000 non-null   int64  \n",
            " 12  Online              5000 non-null   int64  \n",
            " 13  CreditCard          5000 non-null   int64  \n",
            "dtypes: float64(1), int64(13)\n",
            "memory usage: 547.0 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#finding the sum of null values in each column\n",
        "data.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Nk0qITtTzT8-",
        "outputId": "86ab73ee-9110-4577-95ed-4fa651f421cc"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "ID                    0\n",
              "Age                   0\n",
              "Experience            0\n",
              "Income                0\n",
              "ZIP Code              0\n",
              "Family                0\n",
              "CCAvg                 0\n",
              "Education             0\n",
              "Mortgage              0\n",
              "Personal Loan         0\n",
              "Securities Account    0\n",
              "CD Account            0\n",
              "Online                0\n",
              "CreditCard            0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['ID'] = data['ID'].fillna(data['ID'].mode()[0])"
      ],
      "metadata": {
        "id": "XJAowgsXzWaJ"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data['Age'] = data['Age'].fillna(data['Age'].mode()[0])"
      ],
      "metadata": {
        "id": "Dy5laxMIzZ2-"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data['Experience'] = data['Experience'].fillna(data['Experience'].mode()[0])"
      ],
      "metadata": {
        "id": "UUV2YoSdzfpg"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#data['Income']=data['Income'].str.replace('+','')"
      ],
      "metadata": {
        "id": "UCi1ixuHzhn8"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data['ZIP Code'] = data['ZIP Code'].fillna(data['ZIP Code'].mode()[0])\n",
        "data['Family'] = data['Family'].fillna(data['Family'].mode()[0])\n",
        "data['CCAvg'] = data['CCAvg'].fillna(data['CCAvg'].mode()[0])\n",
        "data['Education'] = data['Education'].fillna(data['Education'].mode()[0])\n",
        "data['Mortgage'] = data['Mortgage'].fillna(data['Mortgage'].mode()[0])\n",
        "data['Personal Loan'] = data['Personal Loan'].fillna(data['Personal Loan'].mode()[0])\n",
        "data['Securities Account'] = data['Securities Account'].fillna(data['Securities Account'].mode()[0])\n",
        "data['CD Account'] = data['CD Account'].fillna(data['CD Account'].mode()[0])\n",
        "data['Online'] = data['Online'].fillna(data['Online'].mode()[0])\n",
        "data['CreditCard'] = data['CreditCard'].fillna(data['CreditCard'].mode()[0])"
      ],
      "metadata": {
        "id": "NBxIj1xyzjxh"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data['ID']=data['ID'].astype('int64')\n",
        "data['Age']=data['Age'].astype('int64')\n",
        "data['Experience']=data['Experience'].astype('int64')\n",
        "data['ZIP Code']=data['ZIP Code'].astype('int64')\n",
        "data['Family']=data['Family'].astype('int64')\n",
        "data['Education']=data['Education'].astype('int64')\n",
        "data['Mortgage']=data['Mortgage'].astype('int64')\n",
        "data['Personal Loan']=data['Personal Loan'].astype('int64')\n",
        "data['Securities Account']=data['Securities Account'].astype('int64')\n",
        "data['CD Account']=data['CD Account'].astype('int64')\n",
        "data['Online']=data['Online'].astype('int64')\n",
        "data['CreditCard']=data['CreditCard'].astype('int64')"
      ],
      "metadata": {
        "id": "NI5eYA_nznm5"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Balancine dataset by using smote\n",
        "from imblearn.combine import SMOTETomek"
      ],
      "metadata": {
        "id": "d3RSD6lhztZg"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "smote = SMOTETomek(0.90)"
      ],
      "metadata": {
        "id": "1qPLKk3nzvL4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#dividing the dataset into dependent nd independent y and x respectively\n",
        "y=data['Personal Loan']\n",
        "x=data.drop(columns=['Personal Loan'],axis=1)"
      ],
      "metadata": {
        "id": "Ufvew1LKzxxb"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#ceating a new x and y variables for the balanced set\n",
        "x_bal,y_bal=smote.fit_resample(x,y)"
      ],
      "metadata": {
        "id": "xY1ivfmzz0P4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#printing the values of y before balancing the data and after\n",
        "print(y.value_counts())\n",
        "print(y_bal.value_counts())"
      ],
      "metadata": {
        "id": "X5CGiUrXz2t1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Task 3** Milestone 3"
      ],
      "metadata": {
        "id": "U2kKrjvqz7p6"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 381
        },
        "id": "dk2ML2ktOYzi",
        "outputId": "77b50962-2d95-487f-bd5e-90c86b1718a3"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                ID          Age   Experience       Income      ZIP Code  \\\n",
              "count  5000.000000  5000.000000  5000.000000  5000.000000   5000.000000   \n",
              "mean   2500.500000    45.338400    20.104600    73.774200  93152.503000   \n",
              "std    1443.520003    11.463166    11.467954    46.033729   2121.852197   \n",
              "min       1.000000    23.000000    -3.000000     8.000000   9307.000000   \n",
              "25%    1250.750000    35.000000    10.000000    39.000000  91911.000000   \n",
              "50%    2500.500000    45.000000    20.000000    64.000000  93437.000000   \n",
              "75%    3750.250000    55.000000    30.000000    98.000000  94608.000000   \n",
              "max    5000.000000    67.000000    43.000000   224.000000  96651.000000   \n",
              "\n",
              "            Family        CCAvg    Education     Mortgage  Personal Loan  \\\n",
              "count  5000.000000  5000.000000  5000.000000  5000.000000    5000.000000   \n",
              "mean      2.396400     1.937938     1.881000    56.498800       0.096000   \n",
              "std       1.147663     1.747659     0.839869   101.713802       0.294621   \n",
              "min       1.000000     0.000000     1.000000     0.000000       0.000000   \n",
              "25%       1.000000     0.700000     1.000000     0.000000       0.000000   \n",
              "50%       2.000000     1.500000     2.000000     0.000000       0.000000   \n",
              "75%       3.000000     2.500000     3.000000   101.000000       0.000000   \n",
              "max       4.000000    10.000000     3.000000   635.000000       1.000000   \n",
              "\n",
              "       Securities Account  CD Account       Online   CreditCard  \n",
              "count         5000.000000  5000.00000  5000.000000  5000.000000  \n",
              "mean             0.104400     0.06040     0.596800     0.294000  \n",
              "std              0.305809     0.23825     0.490589     0.455637  \n",
              "min              0.000000     0.00000     0.000000     0.000000  \n",
              "25%              0.000000     0.00000     0.000000     0.000000  \n",
              "50%              0.000000     0.00000     1.000000     0.000000  \n",
              "75%              0.000000     0.00000     1.000000     1.000000  \n",
              "max              1.000000     1.00000     1.000000     1.000000  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b7f9e2f8-a678-4a8e-9c30-60dffaa673df\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>ID</th>\n",
              "      <th>Age</th>\n",
              "      <th>Experience</th>\n",
              "      <th>Income</th>\n",
              "      <th>ZIP Code</th>\n",
              "      <th>Family</th>\n",
              "      <th>CCAvg</th>\n",
              "      <th>Education</th>\n",
              "      <th>Mortgage</th>\n",
              "      <th>Personal Loan</th>\n",
              "      <th>Securities Account</th>\n",
              "      <th>CD Account</th>\n",
              "      <th>Online</th>\n",
              "      <th>CreditCard</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.00000</td>\n",
              "      <td>5000.000000</td>\n",
              "      <td>5000.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>2500.500000</td>\n",
              "      <td>45.338400</td>\n",
              "      <td>20.104600</td>\n",
              "      <td>73.774200</td>\n",
              "      <td>93152.503000</td>\n",
              "      <td>2.396400</td>\n",
              "      <td>1.937938</td>\n",
              "      <td>1.881000</td>\n",
              "      <td>56.498800</td>\n",
              "      <td>0.096000</td>\n",
              "      <td>0.104400</td>\n",
              "      <td>0.06040</td>\n",
              "      <td>0.596800</td>\n",
              "      <td>0.294000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>1443.520003</td>\n",
              "      <td>11.463166</td>\n",
              "      <td>11.467954</td>\n",
              "      <td>46.033729</td>\n",
              "      <td>2121.852197</td>\n",
              "      <td>1.147663</td>\n",
              "      <td>1.747659</td>\n",
              "      <td>0.839869</td>\n",
              "      <td>101.713802</td>\n",
              "      <td>0.294621</td>\n",
              "      <td>0.305809</td>\n",
              "      <td>0.23825</td>\n",
              "      <td>0.490589</td>\n",
              "      <td>0.455637</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>23.000000</td>\n",
              "      <td>-3.000000</td>\n",
              "      <td>8.000000</td>\n",
              "      <td>9307.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.00000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>1250.750000</td>\n",
              "      <td>35.000000</td>\n",
              "      <td>10.000000</td>\n",
              "      <td>39.000000</td>\n",
              "      <td>91911.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.700000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.00000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>2500.500000</td>\n",
              "      <td>45.000000</td>\n",
              "      <td>20.000000</td>\n",
              "      <td>64.000000</td>\n",
              "      <td>93437.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>1.500000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.00000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>3750.250000</td>\n",
              "      <td>55.000000</td>\n",
              "      <td>30.000000</td>\n",
              "      <td>98.000000</td>\n",
              "      <td>94608.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>2.500000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>101.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.00000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>5000.000000</td>\n",
              "      <td>67.000000</td>\n",
              "      <td>43.000000</td>\n",
              "      <td>224.000000</td>\n",
              "      <td>96651.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>10.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>635.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.00000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b7f9e2f8-a678-4a8e-9c30-60dffaa673df')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b7f9e2f8-a678-4a8e-9c30-60dffaa673df button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b7f9e2f8-a678-4a8e-9c30-60dffaa673df');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ],
      "source": [
        "data.describe()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#plotting the using distplot\n",
        "plt.figure(figsize=(12,5))\n",
        "plt.subplot(121)\n",
        "plt.displot(data['Income'])\n",
        "plt.subplot(122)\n",
        "sns.displot(data['CD Account'])\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "_mw05Jo8P86S"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#plotting the count plot\n",
        "plt.figure(figsize=(18,4))\n",
        "plt.subplot(1,4,1)\n",
        "sns.countplot(data['Experience'])\n",
        "plt.subplot(1,4,3)\n",
        "sns.countplot(data['Education'])\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 275
        },
        "id": "OoZc-w03TCVv",
        "outputId": "9f647820-5628-41a1-be71-d7a269e3cb22"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1800x400 with 2 Axes>"
            ],
            "image/png": "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NGlS7rvvvpx//vlJkueffz4nnXRS2tvbM3v27DzyyCM599xzs23bttTX1ydJVqxYkSVLlmTHjh2prq7OkiVLsmrVqjz77LOVY1xwwQXp6urK6tWr91vDrl27smvXrsrfPT09mTZtWrq7u1NbW3sQz37/Zi6+Z9CPCRRr/Y3zSjluT09P6urqSrt+AVA+/fXr6a9h+Cujvz6Q3nrIvLNlz549+eY3v5mdO3emsbEx69evz+7duzNnzpzKmhNPPDHHHnts2tvbkyTt7e05+eSTKzeCJGlqakpPT08lvW9vb++3j31r9u1jf5YvX566urrKNm3atCJPFQAADjr9NUB5Sg9bNm7cmAkTJqSmpiaXXXZZvvOd72TGjBnp6OhIdXV1Jk6c2G99fX19Ojo6kiQdHR39bgT75vfNvdGanp6evPzyy/utaenSpenu7q5sW7duLeJUAQDgoNNfA5RvXNkFnHDCCdmwYUO6u7vzrW99K/Pnz09bW1upNdXU1KSmpqbUGgAAYCD01wDlKz1sqa6uzvHHH58kmTlzZp555pnceuut+cQnPpHe3t50dXX1S987OzvT0NCQJGloaMjTTz/db3/73qb+2jW//4b1zs7O1NbWZvz48QfrtAAAoBT6a4Dylf41ot+3d+/e7Nq1KzNnzswhhxySdevWVeY2bdqULVu2pLGxMUnS2NiYjRs3Zvv27ZU1a9euTW1tbWbMmFFZ89p97Fuzbx8AADCS6a8BBl+pT7YsXbo055xzTo499tj89re/zX333ZfHH388a9asSV1dXRYsWJDW1tYceeSRqa2tzZVXXpnGxsbMnj07SXL22WdnxowZueiii3LDDTeko6MjV199dVpaWiqPKV522WW5/fbbc9VVV+Xiiy/OY489lgceeCCrVq0q89QBAKBw+muAoWFAT7acddZZ6erqet14T09PzjrrrDe9n+3bt2fevHk54YQT8qEPfSjPPPNM1qxZkz/7sz9Lktx8880599xzM3fu3JxxxhlpaGjIt7/97crnx44dm4cffjhjx45NY2NjPvWpT2XevHm5/vrrK2umT5+eVatWZe3atTnllFNy00035c4770xTU9NATh0AAAqnvwYYWar6+vr6DvRDY8aMSUdHRyZPntxvfPv27Xn729+e3bt3F1bgUHAgv6V9MMxcfM+gHxMo1vob55Vy3LKvXwC8OfrrwaW/huGvjP76QK5dB/Q1op/+9KeVf//sZz+r/PxbkuzZsyerV6/O29/+9gMsFwAARif9NcDIdEBhy6mnnpqqqqpUVVXt93HG8ePH58tf/nJhxQEAwEimvwYYmQ4obNm8eXP6+vryzne+M08//XQmTZpUmauurs7kyZMzduzYwosEAICRSH8NMDIdUNhy3HHHJfm/n48DAADeGv01wMg04J9+fuGFF/L9738/27dvf93NYdmyZW+5MAAAGE301wAjx4DClm984xu5/PLLc/TRR6ehoSFVVVWVuaqqKjcDAAA4APprgJFlQGHLF77whfzDP/xDlixZUnQ9AAAw6uivAUaWMQP50G9+85t8/OMfL7oWAAAYlfTXACPLgMKWj3/843n00UeLrgUAAEYl/TXAyDKgrxEdf/zx+fznP5+nnnoqJ598cg455JB+83/zN39TSHEAADAa6K8BRpYBhS1f//rXM2HChLS1taWtra3fXFVVlZsBAAAcAP01wMgyoLBl8+bNRdcBAACjlv4aYGQZ0DtbAAAAANi/AT3ZcvHFF7/h/F133TWgYgAAYDTSXwOMLAMKW37zm9/0+3v37t159tln09XVlbPOOquQwgAAYLTQXwOMLAMKW77zne+8bmzv3r25/PLL80d/9EdvuSgAABhN9NcAI0th72wZM2ZMWltbc/PNNxe1SwAAGLX01wDDV6EvyP3FL36RV199tchdAgDAqKW/BhieBvQ1otbW1n5/9/X15cUXX8yqVasyf/78QgoDAIDRQn8NMLIMKGz5z//8z35/jxkzJpMmTcpNN930B9+kDgAA9Ke/BhhZBhS2fP/73y+6DgAAGLX01wAjy4DCln127NiRTZs2JUlOOOGETJo0qZCiAABgNNJfA4wMA3pB7s6dO3PxxRfnmGOOyRlnnJEzzjgjU6ZMyYIFC/K73/2u6BoBAGBE018DjCwDCltaW1vT1taWhx56KF1dXenq6sr3vve9tLW15bOf/WzRNQIAwIimvwYYWQb0NaJ/+7d/y7e+9a188IMfrIx95CMfyfjx4/OXf/mX+epXv1pUfQAAMOLprwFGlgE92fK73/0u9fX1rxufPHmyxxwBAOAA6a8BRpYBhS2NjY255ppr8sorr1TGXn755Vx33XVpbGwsrDgAABgN9NcAI8uAvkZ0yy235MMf/nCmTp2aU045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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#visualising two columns against each other\n",
        "plt.figure(figure=(20,5))\n",
        "plt.subplot(131)\n",
        "sns.countplot(data['ID'], hue=data['Age'])\n",
        "plt.subplot(132)\n",
        "sns.countplot(data['Experience'], hue=data['Education'])\n",
        "plt.subplot(133)\n",
        "sns.countplot(data['Income'], hue=data['Personal Loan'])"
      ],
      "metadata": {
        "id": "9EO7UTOweqf4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#visulized based gender and income what would be the application status\n",
        "sns.swarmplot(data['Age'],data['Income'], hue = data['CD Account'])"
      ],
      "metadata": {
        "id": "b5qMt_dbhHb7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#perfroming feature scaling iperation using standard scaller on x part of the dataset because\n",
        "#there different type of values in the columns\n",
        "from imblearn.combine import SMOTETomek\n",
        "smote = SMOTETomek(0.90)\n",
        "x_bal,y_bal = smote.fit_resample(x,y)\n",
        "sc=StandardScaler()\n",
        "x_bal=sc.fit_transform(x_bal)\n",
        "x_bal=pd.DataFrame(x.bal,columns=names)"
      ],
      "metadata": {
        "id": "Fiq1VD-Gik__"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#splitting the dataset in train and test on balanced datasew\n",
        "x_train, x_test, y_train, y_test = train_test_split(x_bal, y_bal, test_size=0.33, random_state=42)"
      ],
      "metadata": {
        "id": "Onj7KdEjjhWz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Task 4** Milestone 4"
      ],
      "metadata": {
        "id": "RSP8f64y1T6y"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "JJb7Ib4l0Iyz"
      },
      "outputs": [],
      "source": [
        "def decisionTree(x_train,x_test,y_train,y_test):\n",
        "    dt=DecisionTreeClassifier()\n",
        "    dt.fix(x_train,y_train)\n",
        "    yPred=dt.predict(x_test)\n",
        "    print('***DecisionTreeClassifier***')\n",
        "    print('Confusion matrix')\n",
        "    print(confusion_matrix(y_test,yPred))\n",
        "    print('Classification report')\n",
        "    print(classification_report(y_test,yPred))"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def randomForest(x_train,x_test,y_train,y_test):\n",
        "    rf=RandomForestClassifier()\n",
        "    rf.fit(x_train,y_train)\n",
        "    yPred=rf.predict(x_test)\n",
        "    print('***RandomForestClassifier***')\n",
        "    print('Confusion matrix')\n",
        "    print(confusion_matrix(y_test,yPred))\n",
        "    print('Classification report')\n",
        "    print(classification_report(y_test,yPred))"
      ],
      "metadata": {
        "id": "3ogZhv1s8QNy"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def KNN(x_train,x_test,y_train,y_test):\n",
        "    knn=KNeighborsClassifier()\n",
        "    knn.fit(x_train,y_train)\n",
        "    yPred=knn.predict(x_test)\n",
        "    print('***KNeighborsClassifier***')\n",
        "    print('Confusion matrix')\n",
        "    print(confusion_matrix(y_test,yPred))\n",
        "    print('Classification report')\n",
        "    print(classification_report(y_test,yPred))"
      ],
      "metadata": {
        "id": "_ygM_F1f9DGM"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def xgboost(x_train,x_test,y_train,y_test):\n",
        "    xg=GradientBoostingClassifier()\n",
        "    xg.fit(x_train,y_train)\n",
        "    yPred=xg.predict(x_test)\n",
        "    print('***GradientBoostingClassifier***')\n",
        "    print('Confusion matrix')\n",
        "    print(confusion_matrix(y_test,yPred))\n",
        "    print('Classification report')\n",
        "    print(classification_report(y_test,yPred))"
      ],
      "metadata": {
        "id": "n_ywmi-j9gQL"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#ANN\n",
        "#importing the keras libraries and packages\n",
        "import tensorflow\n",
        "from tensorflow.keras.models import sequential\n",
        "from tensorflow.keras.layers import Dense"
      ],
      "metadata": {
        "id": "Pyd_giwv-ISJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Initialising the ANN\n",
        "classifier = sequential()"
      ],
      "metadata": {
        "id": "s0OaLT7j_Naf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Adding the input layer and the first hidden layer\n",
        "classifier.add(Dense(units=100,activation='relu',input_dim=11))"
      ],
      "metadata": {
        "id": "rRyoE8Hs_mt_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Adding the second hidden layer\n",
        "classifier.add(Dense(units=50,activation='relu'))"
      ],
      "metadata": {
        "id": "MEZDMVbtAH4E"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Adding the output layer\n",
        "classifier.add(Dense(units=1,activation='sigmoid'))"
      ],
      "metadata": {
        "id": "q50Cx0A2Ae1c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Compiling the ANN\n",
        "classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "PynUSBMmAyXL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Fitting the ANN to the Training set\n",
        "model_history=classifier.fit(x.train,y_train,batch_size=100,validation_split=0.2,epochs=100)"
      ],
      "metadata": {
        "id": "hQWs_TAxBR6L"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Gender Married Dependents Education Self Employed Application CoapplicationIncome Loan_AccountTera\n",
        "dtr.predict([[1,1,0,4276,240,0,1]])"
      ],
      "metadata": {
        "id": "2-6X0DL92aij"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rfr.predict([[1,1,0,1,1,4276,1542,145,0,1]])"
      ],
      "metadata": {
        "id": "Gt1y4Bx83Any"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "knn.predict([[1,1,0,1,1,4276,1542,145,0,1]])"
      ],
      "metadata": {
        "id": "Yb-L6p1o3hLc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "xgb.predict([[1,1,0,1,1,4276,1542,145,0,1]])"
      ],
      "metadata": {
        "id": "m5SCXGpH37zG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "classifier.save(\"loan.h5\")"
      ],
      "metadata": {
        "id": "pdQM8C364a7g"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#predicting the Test set results\n",
        "y_pred=classifier.predict(x_test)"
      ],
      "metadata": {
        "id": "4IbDVSN54mQn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred"
      ],
      "metadata": {
        "id": "2GQObQdc4x9Q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred=(y_pred>0.5)\n",
        "y_pred"
      ],
      "metadata": {
        "id": "Jd1gqhAA42CA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def predict_exit(sample_value):\n",
        "sample_value = np.array(sample_value)\n",
        "sample_value=sample_value.reshape(1, -1)\n",
        "sample_value=sc.transform(sample_value)\n",
        "return classifier.predict(sample_value)"
      ],
      "metadata": {
        "id": "CFpucnl25C_l"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_value=[[1,1, 0, 1, 4276, 1542,145,240,0,1]]\n",
        "if predict_exit(sample_value)>0.5:\n",
        "print('prediction: High chance of Loan Approval!')\n",
        "else:\n",
        "print('prediction:low chance Loan Approval.')"
      ],
      "metadata": {
        "id": "YJDSGFfB8CkB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_value=[[1,0, 1, 1, 1, 45, 14,45,240, 1,1]]\n",
        "if predict_exit(sample_value)>0.5:\n",
        "print('prediction: High chance of Loan Approval!')\n",
        "else:\n",
        "print('prediction:low chance Loan Approval.')"
      ],
      "metadata": {
        "id": "AX9wq-k18Fw0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "***Task 5*** Milestone 5"
      ],
      "metadata": {
        "id": "EYZSQnXk8K93"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def compareModel(x_train,x_test,y_train,y_test):\n",
        "decisionTree(x_train,x_test,y_train,y_test)\n",
        "print('-'*100)\n",
        "RandomForest(x_train,x_test,y_train,y_test)\n",
        "print('-'*100)\n",
        "XGB(x_train,x_test,y_train,y_test)\n",
        "print('-'*100)\n",
        "KNN(x_train,x_test,y_train,y_test)\n",
        "print('-'*100)"
      ],
      "metadata": {
        "id": "viD10xON8QZe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "compareModel(X_train,X_test,y_train,y_test)"
      ],
      "metadata": {
        "id": "vGGxwEC78Urd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "yPred = classfier.predict(X_test)\n",
        "print(accuracy_score(y_pred,y_test))\n",
        "print(\"ANN Model\")\n",
        "print(\"Confusion_Matrix\")\n",
        "print(confusion _matrix(y_test,y_pred))\n",
        "print(\"Classification Report\")\n",
        "print(classification_report(y_test,y_pred))"
      ],
      "metadata": {
        "id": "YC0m7dFy8ZJ0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import cros_val_score"
      ],
      "metadata": {
        "id": "9TCtDra48Z51"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Randam forest  model is selected\n",
        "rf = RandomForestClassifier()\n",
        "rf.fit(x_train,y_train)\n",
        "ypred = rf.predict(x_test)"
      ],
      "metadata": {
        "id": "xu9hWTzW8b65"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "f1_score(ypred,y_test,average='weighted')"
      ],
      "metadata": {
        "id": "UHpNQ1dF8gM0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cv = cross_val_score(rf,x,y,cv=5)"
      ],
      "metadata": {
        "id": "gSs5JSt78lMV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "np.mean(cv)"
      ],
      "metadata": {
        "id": "w-CjegTK8pJC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Task 6** Milestone 6"
      ],
      "metadata": {
        "id": "YtauxXjr9eoQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pickle.dump(model,open('rdf.pkl','wb'))"
      ],
      "metadata": {
        "id": "s6GRSlND8si6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from flask import Flask , render_templete, request\n",
        "import numpy as np\n",
        "import pickle"
      ],
      "metadata": {
        "id": "Ne7KfOT78uxG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "app = Flask(_name_)\n",
        "model = pickle.load(open(r'rdf.pkl','rb'))\n",
        "scale = pickle.load(open(r'scole1.pkl','rb'))"
      ],
      "metadata": {
        "id": "oxb3SeCM8yGj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "input_feature=[int(x) for x in request.form.values() ]"
      ],
      "metadata": {
        "id": "a5YsHNJP83OY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "input_feature=[np.array(input_feature)]\n",
        "print(input_features)\n",
        "names = ['Gender','Married','Depends','Education','ApplicantIncome',CoapplicatntIncome','LoanAmount','Loan_Amount_Term','Crdit_history','Property_Area']\n",
        "data = pandas.DataFrame(input_features,columns=names\n",
        "print(data)"
      ],
      "metadata": {
        "id": "WvUZ_NWV8_mO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prediction = model1.predict(data)\n",
        "print(prediction)\n",
        "prediction=int(prediction)\n",
        "print(type(prediction))"
      ],
      "metadata": {
        "id": "0h6HrOb29DEv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "if (prediction == 0):\n",
        "return render_template(\"output.html\",result=\"Loan Will not be Approved\")\n",
        "else:\n",
        "return render_template(\"output.html\",result=\"Loan Will not be Approved\")"
      ],
      "metadata": {
        "id": "Hp3qAS5Z9FcY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__==\"__main__\":"
      ],
      "metadata": {
        "id": "T2DZ5L_M9JH3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "port=int(os.environ.get('PORT',5000))\n",
        "app.run(debug=false)"
      ],
      "metadata": {
        "id": "WnSTFw9t9Nih"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "pehiS2P29QGs"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}