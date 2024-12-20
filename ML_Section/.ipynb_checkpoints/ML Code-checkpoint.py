{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5b26338c-5e80-410e-b19a-cc7e2cd1d4e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "# Suppress the FutureWarning\n",
    "warnings.filterwarnings(\"ignore\", category=FutureWarning)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b43fd9ad-2f43-457e-a57b-2c4c20d0d284",
   "metadata": {},
   "source": [
    "# Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8bcf9027-6396-4bb8-91c1-b6e96434aa2a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import essential libraries for data processing, machine learning, visualization, and evaluation\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "299532ee-09f4-4956-ac2a-a0705cb64ad3",
   "metadata": {},
   "source": [
    "# Import Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d22709fa-3fb9-47cc-85bf-09f9317d0ccd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
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
       "      <th>name</th>\n",
       "      <th>year</th>\n",
       "      <th>selling_price</th>\n",
       "      <th>km_driven</th>\n",
       "      <th>fuel</th>\n",
       "      <th>seller_type</th>\n",
       "      <th>transmission</th>\n",
       "      <th>owner</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Maruti 800 AC</td>\n",
       "      <td>2007</td>\n",
       "      <td>60000</td>\n",
       "      <td>70000</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Maruti Wagon R LXI Minor</td>\n",
       "      <td>2007</td>\n",
       "      <td>135000</td>\n",
       "      <td>50000</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Hyundai Verna 1.6 SX</td>\n",
       "      <td>2012</td>\n",
       "      <td>600000</td>\n",
       "      <td>100000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Datsun RediGO T Option</td>\n",
       "      <td>2017</td>\n",
       "      <td>250000</td>\n",
       "      <td>46000</td>\n",
       "      <td>Petrol</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>First Owner</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Honda Amaze VX i-DTEC</td>\n",
       "      <td>2014</td>\n",
       "      <td>450000</td>\n",
       "      <td>141000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>Individual</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Second Owner</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       name  year  selling_price  km_driven    fuel  \\\n",
       "0             Maruti 800 AC  2007          60000      70000  Petrol   \n",
       "1  Maruti Wagon R LXI Minor  2007         135000      50000  Petrol   \n",
       "2      Hyundai Verna 1.6 SX  2012         600000     100000  Diesel   \n",
       "3    Datsun RediGO T Option  2017         250000      46000  Petrol   \n",
       "4     Honda Amaze VX i-DTEC  2014         450000     141000  Diesel   \n",
       "\n",
       "  seller_type transmission         owner  \n",
       "0  Individual       Manual   First Owner  \n",
       "1  Individual       Manual   First Owner  \n",
       "2  Individual       Manual   First Owner  \n",
       "3  Individual       Manual   First Owner  \n",
       "4  Individual       Manual  Second Owner  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_csv('./Car_Details.csv')\n",
    "\n",
    "# Preview dataset\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c7277f94-8a66-4ca9-b6e5-9dbe6aa04f36",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',\n",
      "       'transmission', 'owner'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9804b017-6936-4432-b10b-16f4976a8294",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4340, 8) \n",
      "\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4340 entries, 0 to 4339\n",
      "Data columns (total 8 columns):\n",
      " #   Column         Non-Null Count  Dtype \n",
      "---  ------         --------------  ----- \n",
      " 0   name           4340 non-null   object\n",
      " 1   year           4340 non-null   int64 \n",
      " 2   selling_price  4340 non-null   int64 \n",
      " 3   km_driven      4340 non-null   int64 \n",
      " 4   fuel           4340 non-null   object\n",
      " 5   seller_type    4340 non-null   object\n",
      " 6   transmission   4340 non-null   object\n",
      " 7   owner          4340 non-null   object\n",
      "dtypes: int64(3), object(5)\n",
      "memory usage: 271.4+ KB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(df.shape ,\"\\n\")\n",
    "\n",
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d88ab7ae-99e1-4fc2-9c70-0f11fce91f47",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8900000"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['selling_price'].max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1461e694-0abe-4be6-910e-11738cd8caf1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    4.340000e+03\n",
       "mean     5.041273e+05\n",
       "std      5.785487e+05\n",
       "min      2.000000e+04\n",
       "25%      2.087498e+05\n",
       "50%      3.500000e+05\n",
       "75%      6.000000e+05\n",
       "max      8.900000e+06\n",
       "Name: selling_price, dtype: float64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['selling_price'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "921418f8-7944-471d-9a6b-be0429b03afe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "20000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['selling_price'].min() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "11f69e05-c4d9-4fe2-8da1-de3128864f56",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                       name  year  selling_price  km_driven    fuel  \\\n",
      "0             Maruti 800 AC  2007          60000      70000  Petrol   \n",
      "1  Maruti Wagon R LXI Minor  2007         135000      50000  Petrol   \n",
      "2      Hyundai Verna 1.6 SX  2012         600000     100000  Diesel   \n",
      "3    Datsun RediGO T Option  2017         250000      46000  Petrol   \n",
      "4     Honda Amaze VX i-DTEC  2014         450000     141000  Diesel   \n",
      "\n",
      "  seller_type transmission         owner selling_price_range  \n",
      "0  Individual       Manual   First Owner    Less than 100000  \n",
      "1  Individual       Manual   First Owner    100001 - 1000000  \n",
      "2  Individual       Manual   First Owner    100001 - 1000000  \n",
      "3  Individual       Manual   First Owner    100001 - 1000000  \n",
      "4  Individual       Manual  Second Owner    100001 - 1000000  \n"
     ]
    }
   ],
   "source": [
    "# One-liner using nested if-else to create the 'selling_price_range' column\n",
    "df['selling_price_range'] = df['selling_price'].apply(\n",
    "    lambda x: 'Less than 100000' if x <= 100000 else\n",
    "              '100001 - 1000000' if x <= 1000000 else\n",
    "              '1000001 - 2000000' if x <= 2000000 else\n",
    "              '2000001 - 3000000' if x <= 3000000 else\n",
    "              '3000001 - 4000000' if x <= 4000000 else\n",
    "              '4000001 - 5000000' if x <= 5000000 else\n",
    "              '5000001 - 6000000' if x <= 6000000 else\n",
    "              '6000001 - 7000000' if x <= 7000000 else\n",
    "              '7000001 - 8000000' if x <= 8000000 else\n",
    "              '8000001 - 9000000'\n",
    ")\n",
    "\n",
    "# Display the first few rows of the updated DataFrame with the new classification column\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e40cd4d1-b338-454e-9531-fbab59b61db3",
   "metadata": {},
   "source": [
    "# PreProcessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "70ac090d-23eb-49bc-82f3-7543c8c1aad6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fuel\n",
      "Diesel      2153\n",
      "Petrol      2123\n",
      "CNG           40\n",
      "LPG           23\n",
      "Electric       1\n",
      "Name: count, dtype: int64\n",
      "\n",
      " ============================= \n",
      "\n",
      "seller_type\n",
      "Individual          3244\n",
      "Dealer               994\n",
      "Trustmark Dealer     102\n",
      "Name: count, dtype: int64\n",
      "\n",
      " ============================= \n",
      "\n",
      "transmission\n",
      "Manual       3892\n",
      "Automatic     448\n",
      "Name: count, dtype: int64\n",
      "\n",
      " ============================= \n",
      "\n",
      "owner\n",
      "First Owner             2832\n",
      "Second Owner            1106\n",
      "Third Owner              304\n",
      "Fourth & Above Owner      81\n",
      "Test Drive Car            17\n",
      "Name: count, dtype: int64\n",
      "\n",
      " ============================= \n",
      "\n",
      "selling_price_range\n",
      "100001 - 1000000     3619\n",
      "Less than 100000      379\n",
      "1000001 - 2000000     245\n",
      "2000001 - 3000000      48\n",
      "3000001 - 4000000      26\n",
      "4000001 - 5000000      20\n",
      "8000001 - 9000000       2\n",
      "5000001 - 6000000       1\n",
      "Name: count, dtype: int64\n",
      "\n",
      " ============================= \n",
      "\n"
     ]
    }
   ],
   "source": [
    "## Unique Values and their counts in 'Fuel Type'\n",
    "print(df['fuel'].value_counts())\n",
    "\n",
    "print(\"\\n ============================= \\n\")\n",
    "\n",
    "## Unique Values and their counts in 'Seller Type'\n",
    "print(df['seller_type'].value_counts())\n",
    "\n",
    "print(\"\\n ============================= \\n\")\n",
    "\n",
    "## Unique Values and their counts in 'Transmission Type'\n",
    "print(df['transmission'].value_counts())\n",
    "\n",
    "print(\"\\n ============================= \\n\")\n",
    "\n",
    "## Unique Values and their counts in 'Owner Type'\n",
    "print(df['owner'].value_counts())\n",
    "\n",
    "print(\"\\n ============================= \\n\")\n",
    "\n",
    "## Unique Values and their counts in 'Owner Type'\n",
    "print(df['selling_price_range'].value_counts())\n",
    "\n",
    "print(\"\\n ============================= \\n\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fbd53c1e-def4-4140-bd60-feca4eafe2f9",
   "metadata": {},
   "source": [
    "## Convert Categorical features to Numerical Values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55800076-c382-4df3-94cf-46e239e01428",
   "metadata": {},
   "source": [
    "| Features | Values : replacements |\n",
    "| -------- | --------------------- |\n",
    "| **Fuel** |  Diesel : 1   ;   Petrol : 2   ;   CNG : 3   ;   LPG : 4   ;   Electric : 5  |\n",
    "| **Seller_Type** | Individual : 1   ;   Dealer : 2   ;   Trustmark Dealer : 3 |\n",
    "| **Transmission** | Manual : 1   ;   Automatic : 2 |\n",
    "| **Owner** | First Owner : 1   ;   Second Owner : 2   ;   Third Owner : 3   ;   Fourth & Above Owner : 4   ;   Test Drive Car: 5 |\n",
    "| **Selling_price_range** | Less than 100000 : 1 ; 100001 - 1000000 : 2 ; 1000001 - 2000000 : 3 ; 2000001 - 3000000 : 4 ; 3000001 - 4000000 : 5 ; 4000001 - 5000000 : 6 ; 5000001 - 6000000 : 7 ; 6000001 - 7000000 : 8 ; 7000001 - 8000000 : 9 ; 8000001 - 9000000 : 10 |\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "864a9066-d638-494d-8827-1794d3d9ae25",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=================================================\n",
      "Converted 'fuel' column:\n",
      "fuel\n",
      "1    2153\n",
      "2    2123\n",
      "3      40\n",
      "4      23\n",
      "5       1\n",
      "Name: count, dtype: int64\n",
      "=================================================\n",
      "Converted 'seller_type' column:\n",
      "seller_type\n",
      "1    3244\n",
      "2     994\n",
      "3     102\n",
      "Name: count, dtype: int64\n",
      "=================================================\n",
      "Converted 'transmission' column:\n",
      "transmission\n",
      "1    3892\n",
      "2     448\n",
      "Name: count, dtype: int64\n",
      "=================================================\n",
      "Converted 'owner' column:\n",
      "owner\n",
      "1    2832\n",
      "2    1106\n",
      "3     304\n",
      "4      81\n",
      "5      17\n",
      "Name: count, dtype: int64\n",
      "=================================================\n",
      "Converted 'selling_price_range' column:\n",
      "selling_price_range\n",
      "2     3619\n",
      "1      379\n",
      "3      245\n",
      "4       48\n",
      "5       26\n",
      "6       20\n",
      "10       2\n",
      "7        1\n",
      "Name: count, dtype: int64\n",
      "=================================================\n"
     ]
    }
   ],
   "source": [
    "print(\"=================================================\")\n",
    "\n",
    "# Conversion of 'fuel' column from categorical to numerical using a mapping dictionary\n",
    "mapping = {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5}\n",
    "df['fuel'] = df['fuel'].map(mapping)\n",
    "\n",
    "# Printing the conversion\n",
    "print(\"Converted 'fuel' column:\")\n",
    "print(df['fuel'].value_counts())\n",
    "print(\"=================================================\")\n",
    "\n",
    "# -------------------------------------------------------------------------------------\n",
    "\n",
    "# Conversion of 'fuel' column from categorical to numerical using a mapping dictionary\n",
    "mapping = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}\n",
    "df['seller_type'] = df['seller_type'].map(mapping)\n",
    "\n",
    "# Printing the conversion\n",
    "print(\"Converted 'seller_type' column:\")\n",
    "print(df['seller_type'].value_counts())\n",
    "print(\"=================================================\")\n",
    "\n",
    "# -------------------------------------------------------------------------------------\n",
    "\n",
    "# Conversion of 'fuel' column from categorical to numerical using a mapping dictionary\n",
    "mapping = {'Manual': 1, 'Automatic': 2}\n",
    "df['transmission'] = df['transmission'].map(mapping)\n",
    "\n",
    "# Printing the conversion\n",
    "print(\"Converted 'transmission' column:\")\n",
    "print(df['transmission'].value_counts())\n",
    "print(\"=================================================\")\n",
    "\n",
    "# -------------------------------------------------------------------------------------\n",
    "\n",
    "# Conversion of 'fuel' column from categorical to numerical using a mapping dictionary\n",
    "mapping = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,  'Fourth & Above Owner': 4, 'Test Drive Car': 5 }\n",
    "df['owner'] = df['owner'].map(mapping)\n",
    "\n",
    "# Printing the conversion\n",
    "print(\"Converted 'owner' column:\")\n",
    "print(df['owner'].value_counts())\n",
    "print(\"=================================================\")\n",
    "\n",
    "# -------------------------------------------------------------------------------------\n",
    "\n",
    "# Conversion of 'fuel' column from categorical to numerical using a mapping dictionary\n",
    "mapping = {\n",
    "    'Less than 100000' : 1 ,\n",
    "    '100001 - 1000000' : 2 , \n",
    "    '1000001 - 2000000' : 3 , \n",
    "    '2000001 - 3000000' : 4 , \n",
    "    '3000001 - 4000000' : 5 , \n",
    "    '4000001 - 5000000' : 6 , \n",
    "    '5000001 - 6000000' : 7 , \n",
    "    '6000001 - 7000000' : 8 , \n",
    "    '7000001 - 8000000' : 9 , \n",
    "    '8000001 - 9000000' : 10 ,\n",
    "}\n",
    "df['selling_price_range'] = df['selling_price_range'].map(mapping)\n",
    "\n",
    "# Printing the conversion\n",
    "print(\"Converted 'selling_price_range' column:\")\n",
    "print(df['selling_price_range'].value_counts())\n",
    "print(\"=================================================\")\n",
    "\n",
    "\n",
    "\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "da44eb23-7395-47a9-a864-26bb651292eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
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
       "      <th>name</th>\n",
       "      <th>year</th>\n",
       "      <th>selling_price</th>\n",
       "      <th>km_driven</th>\n",
       "      <th>fuel</th>\n",
       "      <th>seller_type</th>\n",
       "      <th>transmission</th>\n",
       "      <th>owner</th>\n",
       "      <th>selling_price_range</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Maruti 800 AC</td>\n",
       "      <td>2007</td>\n",
       "      <td>60000</td>\n",
       "      <td>70000</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Maruti Wagon R LXI Minor</td>\n",
       "      <td>2007</td>\n",
       "      <td>135000</td>\n",
       "      <td>50000</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Hyundai Verna 1.6 SX</td>\n",
       "      <td>2012</td>\n",
       "      <td>600000</td>\n",
       "      <td>100000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Datsun RediGO T Option</td>\n",
       "      <td>2017</td>\n",
       "      <td>250000</td>\n",
       "      <td>46000</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Honda Amaze VX i-DTEC</td>\n",
       "      <td>2014</td>\n",
       "      <td>450000</td>\n",
       "      <td>141000</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       name  year  selling_price  km_driven  fuel  \\\n",
       "0             Maruti 800 AC  2007          60000      70000     2   \n",
       "1  Maruti Wagon R LXI Minor  2007         135000      50000     2   \n",
       "2      Hyundai Verna 1.6 SX  2012         600000     100000     1   \n",
       "3    Datsun RediGO T Option  2017         250000      46000     2   \n",
       "4     Honda Amaze VX i-DTEC  2014         450000     141000     1   \n",
       "\n",
       "   seller_type  transmission  owner  selling_price_range  \n",
       "0            1             1      1                    1  \n",
       "1            1             1      1                    2  \n",
       "2            1             1      1                    2  \n",
       "3            1             1      1                    2  \n",
       "4            1             1      2                    2  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c9074db-3e9c-453b-9de1-b8c2540e24d5",
   "metadata": {},
   "source": [
    "# Selecting Features (Inputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "176080c1-2619-4ffd-9f3e-2abaa597dae8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Features(X) \n",
    "X = df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8790290a-721c-40f6-bec2-32ffb945ec51",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4340 entries, 0 to 4339\n",
      "Data columns (total 6 columns):\n",
      " #   Column        Non-Null Count  Dtype\n",
      "---  ------        --------------  -----\n",
      " 0   year          4340 non-null   int64\n",
      " 1   km_driven     4340 non-null   int64\n",
      " 2   fuel          4340 non-null   int64\n",
      " 3   seller_type   4340 non-null   int64\n",
      " 4   transmission  4340 non-null   int64\n",
      " 5   owner         4340 non-null   int64\n",
      "dtypes: int64(6)\n",
      "memory usage: 203.6 KB\n"
     ]
    }
   ],
   "source": [
    "X.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ce50adf-d18f-4214-9891-6b51fe04d045",
   "metadata": {},
   "source": [
    "# Selecting Label (Output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "57ac7dad-22ef-4ddb-b5d0-841693d5f8f4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     60000\n",
       "1    135000\n",
       "2    600000\n",
       "3    250000\n",
       "4    450000\n",
       "Name: selling_price, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y = df['selling_price'] \n",
    "Y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "9621a8c5-cc86-4030-9301-bf9afb505f6d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    1\n",
       "1    2\n",
       "2    2\n",
       "3    2\n",
       "4    2\n",
       "Name: selling_price_range, dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_classification = df['selling_price_range'] \n",
    "Y_classification.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6164d995-2c24-40a3-98fd-76804006abe8",
   "metadata": {},
   "source": [
    "## Normalize Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "cbe883b8-35a4-4602-8db2-02521fb12198",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-1.44507431,  0.08113906,  0.86482829, -0.5555905 , -0.33927557,\n",
       "        -0.63031847],\n",
       "       [-1.44507431, -0.3476891 ,  0.86482829, -0.5555905 , -0.33927557,\n",
       "        -0.63031847],\n",
       "       [-0.2587948 ,  0.7243813 , -0.95365755, -0.5555905 , -0.33927557,\n",
       "        -0.63031847],\n",
       "       [ 0.92748471, -0.43345473,  0.86482829, -0.5555905 , -0.33927557,\n",
       "        -0.63031847],\n",
       "       [ 0.215717  ,  1.60347903, -0.95365755, -0.5555905 , -0.33927557,\n",
       "         0.7205863 ]])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn import preprocessing\n",
    "\n",
    "X = preprocessing.StandardScaler().fit(X).transform(X)\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "426c9a46-5c8a-4439-bfea-cdff70e71f20",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 17,  60, 236, 114, 193], dtype=int64)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Assuming y is your target label\n",
    "label_encoder = LabelEncoder()\n",
    "y = label_encoder.fit_transform(Y)\n",
    "y[0:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b07477a5-9664-40fd-ad57-4e8133e6e669",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 1, 1, 1], dtype=int64)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Assuming y is your target label\n",
    "label_encoder = LabelEncoder()\n",
    "y_classification = label_encoder.fit_transform(Y_classification)\n",
    "y_classification[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aca50516-8f49-4cda-96a1-819821687a0c",
   "metadata": {},
   "source": [
    "## Split the Data into Training and Testing Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "e11c0758-f08a-4d9b-b696-157296222b5a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train set: (3472, 6) (3472,)\n",
      "Test set: (868, 6) (868,)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split( X, \n",
    "                                                    y, \n",
    "                                                    test_size = 0.2,\n",
    "                                                    random_state = 10)\n",
    "print ('Train set:', X_train.shape, y_train.shape)\n",
    "print ('Test set:', X_test.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "6dae761b-9a95-4472-b2a4-c99c4250fa1a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train set: (3472, 6) (3472,)\n",
      "Test set: (868, 6) (868,)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_classification_train, y_classification_test = train_test_split( X, \n",
    "                                                    y_classification, \n",
    "                                                    test_size = 0.2,\n",
    "                                                    random_state = 10)\n",
    "print ('Train set:', X_train.shape, y_classification_train.shape)\n",
    "print ('Test set:', X_test.shape, y_classification_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "16c0f343-5cc0-43db-9df1-3f5c01d34028",
   "metadata": {},
   "source": [
    "# Model Building"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "efc1b252-896e-45f0-91fe-ab197c66803b",
   "metadata": {},
   "source": [
    "# Evaluating the Best Model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "260ba097-f84c-4cba-89a4-088762b9dd35",
   "metadata": {},
   "source": [
    "## 1. Regression\n",
    "\n",
    "**Algorithms I'm using**\n",
    "\n",
    "A) Linear Regression \n",
    "\n",
    "B) Random Forest Regressor\n",
    "\n",
    "C) Gradient Boosting Regressor\n",
    "\n",
    "D) XGBoost Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "9c93c67b-d235-4375-8b70-1233ec45589b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train set: (3472, 6) (3472,)\n",
      "Test set: (868, 6) (868,)\n",
      "Fitting 5 folds for each of 2 candidates, totalling 10 fits\n",
      "Linear Regression Best Params: {'fit_intercept': True}\n",
      "Linear Regression R-squared Score: 0.6260495361618581\n",
      "Fitting 5 folds for each of 6 candidates, totalling 30 fits\n",
      "Random Forest Regressor Best Params: {'max_depth': 10, 'n_estimators': 100}\n",
      "Random Forest Regressor R-squared Score: 0.6918184367045845\n",
      "Fitting 5 folds for each of 8 candidates, totalling 40 fits\n",
      "Gradient Boosting Regressor Best Params: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}\n",
      "Gradient Boosting Regressor R-squared Score: 0.6979828214528354\n",
      "Fitting 5 folds for each of 8 candidates, totalling 40 fits\n",
      "XGBoost Regressor Best Params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}\n",
      "XGBoost Regressor R-squared Score: 0.6846190469114681\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.ensemble import GradientBoostingRegressor\n",
    "from xgboost import XGBRegressor\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Dictionary to store the evaluation results\n",
    "regression_results = {}\n",
    "\n",
    "# Function to evaluate a regression model\n",
    "def evaluate_regression_model(model, param_grid, X_train, X_test, y_train, y_test, model_name):\n",
    "    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)\n",
    "    grid_search.fit(X_train, y_train)\n",
    "    \n",
    "    best_model = grid_search.best_estimator_\n",
    "    y_pred = best_model.predict(X_test)\n",
    "    \n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    \n",
    "    # Store results\n",
    "    regression_results[model_name] = {\n",
    "        'Best Params': grid_search.best_params_,\n",
    "        'R-squared Score': r2\n",
    "    }\n",
    "    print(f\"{model_name} Best Params: {grid_search.best_params_}\")\n",
    "    print(f\"{model_name} R-squared Score: {r2}\")\n",
    "\n",
    "# Example parameters for regression models\n",
    "X_train, X_test, y_train, y_test = train_test_split( X, \n",
    "                                                    y, \n",
    "                                                    test_size = 0.2,\n",
    "                                                    random_state = 10)\n",
    "print ('Train set:', X_train.shape, y_train.shape)\n",
    "print ('Test set:', X_test.shape, y_test.shape)\n",
    "\n",
    "# Linear Regression\n",
    "evaluate_regression_model(\n",
    "    LinearRegression(), \n",
    "    param_grid={'fit_intercept': [True, False]},\n",
    "    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,\n",
    "    model_name='Linear Regression'\n",
    ")\n",
    "\n",
    "# Random Forest Regressor\n",
    "evaluate_regression_model(\n",
    "    RandomForestRegressor(), \n",
    "    param_grid={'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},\n",
    "    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,\n",
    "    model_name='Random Forest Regressor'\n",
    ")\n",
    "\n",
    "# Gradient Boosting Regressor\n",
    "evaluate_regression_model(\n",
    "    GradientBoostingRegressor(), \n",
    "    param_grid={'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},\n",
    "    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,\n",
    "    model_name='Gradient Boosting Regressor'\n",
    ")\n",
    "\n",
    "# XGBoost Regressor\n",
    "evaluate_regression_model(\n",
    "    XGBRegressor(), \n",
    "    param_grid={'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},\n",
    "    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,\n",
    "    model_name='XGBoost Regressor'\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ad31603-91dd-4089-bb08-7cb40795abf0",
   "metadata": {},
   "source": [
    "## 2. Classification\n",
    "\n",
    "**Algorithms I'm using**\n",
    "\n",
    "A) SVM Classifier\n",
    "\n",
    "B) Decision Tree Classifier\n",
    "\n",
    "C) Random Forest Classifier\n",
    "\n",
    "D) XGBoost Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "beddd324-ab07-4f4a-92e5-4469d32cd298",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train set: (3472, 6) (3472,)\n",
      "Test set: (868, 6) (868,)\n",
      "Fitting 5 folds for each of 12 candidates, totalling 60 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\model_selection\\_split.py:737: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVM Classifier Best Params: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}\n",
      "SVM Classifier F1 Score: 0.87669526272569\n",
      "Fitting 5 folds for each of 6 candidates, totalling 30 fits\n",
      "Decision Tree Classifier Best Params: {'criterion': 'gini', 'max_depth': 10}\n",
      "Decision Tree Classifier F1 Score: 0.8777478622086794\n",
      "Fitting 5 folds for each of 6 candidates, totalling 30 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\model_selection\\_split.py:737: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.\n",
      "  warnings.warn(\n",
      "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\model_selection\\_split.py:737: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Classifier Best Params: {'max_depth': 10, 'n_estimators': 100}\n",
      "Random Forest Classifier F1 Score: 0.8887260767397489\n",
      "Fitting 5 folds for each of 8 candidates, totalling 40 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\model_selection\\_split.py:737: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.\n",
      "  warnings.warn(\n",
      "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\model_selection\\_validation.py:547: FitFailedWarning: \n",
      "8 fits failed out of a total of 40.\n",
      "The score on these train-test partitions for these parameters will be set to nan.\n",
      "If these failures are not expected, you can try to debug them by setting error_score='raise'.\n",
      "\n",
      "Below are more details about the failures:\n",
      "--------------------------------------------------------------------------------\n",
      "8 fits failed with the following error:\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\model_selection\\_validation.py\", line 895, in _fit_and_score\n",
      "    estimator.fit(X_train, y_train, **fit_params)\n",
      "  File \"C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\xgboost\\core.py\", line 726, in inner_f\n",
      "    return func(**kwargs)\n",
      "           ^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\xgboost\\sklearn.py\", line 1491, in fit\n",
      "    raise ValueError(\n",
      "ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5 6], got [0 1 2 3 4 5 7]\n",
      "\n",
      "  warnings.warn(some_fits_failed_message, FitFailedWarning)\n",
      "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\model_selection\\_search.py:1051: UserWarning: One or more of the test scores are non-finite: [nan nan nan nan nan nan nan nan]\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBoost Classifier Best Params: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}\n",
      "XGBoost Classifier F1 Score: 0.8625311777220561\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import f1_score\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Dictionary to store the evaluation results\n",
    "classification_results = {}\n",
    "\n",
    "# Function to evaluate a classification model\n",
    "def evaluate_classification_model(model, param_grid, X_train, X_test, y_classification_train, y_classification_test, model_name):\n",
    "    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)\n",
    "    grid_search.fit(X_train, y_classification_train)\n",
    "    \n",
    "    best_model = grid_search.best_estimator_\n",
    "    y_pred = best_model.predict(X_test)\n",
    "    \n",
    "    f1 = f1_score(y_classification_test, y_pred, average='weighted')\n",
    "    \n",
    "    # Store results\n",
    "    classification_results[model_name] = {\n",
    "        'Best Params': grid_search.best_params_,\n",
    "        'F1 Score': f1\n",
    "    }\n",
    "    print(f\"{model_name} Best Params: {grid_search.best_params_}\")\n",
    "    print(f\"{model_name} F1 Score: {f1}\")\n",
    "\n",
    "# Example parameters for classification models\n",
    "X_train, X_test, y_classification_train, y_classification_test = train_test_split( X, \n",
    "                                                    y_classification, \n",
    "                                                    test_size = 0.2,\n",
    "                                                    random_state = 10)\n",
    "print ('Train set:', X_train.shape, y_classification_train.shape)\n",
    "print ('Test set:', X_test.shape, y_classification_test.shape)\n",
    "\n",
    "# SVM Classifier\n",
    "evaluate_classification_model(\n",
    "    SVC(), \n",
    "    param_grid={'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},\n",
    "    X_train=X_train, X_test=X_test, y_classification_train=y_classification_train, y_classification_test=y_classification_test,\n",
    "    model_name='SVM Classifier'\n",
    ")\n",
    "\n",
    "# Decision Tree Classifier\n",
    "evaluate_classification_model(\n",
    "    DecisionTreeClassifier(), \n",
    "    param_grid={'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20]},\n",
    "    X_train=X_train, X_test=X_test, y_classification_train=y_classification_train, y_classification_test=y_classification_test,\n",
    "    model_name='Decision Tree Classifier'\n",
    ")\n",
    "\n",
    "# Random Forest Classifier\n",
    "evaluate_classification_model(\n",
    "    RandomForestClassifier(), \n",
    "    param_grid={'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},\n",
    "    X_train=X_train, X_test=X_test, y_classification_train=y_classification_train, y_classification_test=y_classification_test,\n",
    "    model_name='Random Forest Classifier'\n",
    ")\n",
    "\n",
    "# XGBoost Classifier\n",
    "evaluate_classification_model(\n",
    "    XGBClassifier(), \n",
    "    param_grid={'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},\n",
    "    X_train=X_train, X_test=X_test, y_classification_train=y_classification_train, y_classification_test=y_classification_test,\n",
    "    model_name='XGBoost Classifier'\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97f0bf4b-8c60-4a48-9adf-e73e131aaef8",
   "metadata": {},
   "source": [
    "## 3. Clustering :\n",
    "\n",
    "**Algorithms I'm using**\n",
    "\n",
    "A) K-Means Clustering\n",
    "\n",
    "B) Hierarchical Clustering\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f7c3d08f-945d-4a81-bd01-93691c2726fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\joblib\\externals\\loky\\backend\\context.py:136: UserWarning: Could not find the number of physical cores for the following reason:\n",
      "found 0 physical cores < 1\n",
      "Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.\n",
      "  warnings.warn(\n",
      "  File \"C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\joblib\\externals\\loky\\backend\\context.py\", line 282, in _count_physical_cores\n",
      "    raise ValueError(f\"found {cpu_count_physical} physical cores < 1\")\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K-Means Clustering Davies-Bouldin Index: 0.5434386313544408\n",
      "Hierarchical Clustering Davies-Bouldin Index: 0.5538945038271399\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import davies_bouldin_score\n",
    "from sklearn.cluster import KMeans, AgglomerativeClustering\n",
    "\n",
    "# Dictionary to store the evaluation results\n",
    "clustering_results = {}\n",
    "\n",
    "# Function to evaluate a clustering model\n",
    "def evaluate_clustering_model(model, X, model_name):\n",
    "    model.fit(X)\n",
    "    labels = model.labels_\n",
    "    \n",
    "    # Calculate Davies-Bouldin Index\n",
    "    db_index = davies_bouldin_score(X, labels)\n",
    "    \n",
    "    # Store results\n",
    "    clustering_results[model_name] = {\n",
    "        'Davies-Bouldin Index': db_index\n",
    "    }\n",
    "    print(f\"{model_name} Davies-Bouldin Index: {db_index}\")\n",
    "\n",
    "# Example data (X for clustering, no need for train-test split)\n",
    "X = df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]\n",
    "\n",
    "# K-Means Clustering\n",
    "evaluate_clustering_model(\n",
    "    KMeans(n_clusters=5), \n",
    "    X=X, \n",
    "    model_name='K-Means Clustering'\n",
    ")\n",
    "\n",
    "# Hierarchical Clustering\n",
    "evaluate_clustering_model(\n",
    "    AgglomerativeClustering(n_clusters=5), \n",
    "    X=X, \n",
    "    model_name='Hierarchical Clustering'\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d97f9726-7659-4371-ae34-3e5684fa267f",
   "metadata": {},
   "source": [
    "## Comparing All Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "6ca93c56-c2f7-47b5-a6d8-ed6748c91d6a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Regression Model: Gradient Boosting Regressor with R-squared Score: 0.6979828214528354\n",
      "Best Classification Model: Random Forest Classifier with F1 Score: 0.8887260767397489\n",
      "Best Clustering Model: K-Means Clustering with Davies-Bouldin Index: 0.5434386313544408\n",
      "\n",
      "\n",
      "Most Suitable Models:\n",
      "('Gradient Boosting Regressor', {'Best Params': {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}, 'R-squared Score': 0.6979828214528354})\n",
      "('Random Forest Classifier', {'Best Params': {'max_depth': 10, 'n_estimators': 100}, 'F1 Score': 0.8887260767397489})\n",
      "('K-Means Clustering', {'Davies-Bouldin Index': 0.5434386313544408})\n"
     ]
    }
   ],
   "source": [
    "# Function to compare and find the best model\n",
    "def compare_models():\n",
    "    # Compare regression models based on R-squared score\n",
    "    best_regression_model = max(regression_results.items(), key=lambda x: x[1]['R-squared Score'])\n",
    "    print(f\"Best Regression Model: {best_regression_model[0]} with R-squared Score: {best_regression_model[1]['R-squared Score']}\")\n",
    "    \n",
    "    # Compare classification models based on F1 Score\n",
    "    best_classification_model = max(classification_results.items(), key=lambda x: x[1]['F1 Score'])\n",
    "    print(f\"Best Classification Model: {best_classification_model[0]} with F1 Score: {best_classification_model[1]['F1 Score']}\")\n",
    "    \n",
    "    # Compare clustering models based on Davies-Bouldin Index (lower is better)\n",
    "    best_clustering_model = min(clustering_results.items(), key=lambda x: x[1]['Davies-Bouldin Index'])\n",
    "    print(f\"Best Clustering Model: {best_clustering_model[0]} with Davies-Bouldin Index: {best_clustering_model[1]['Davies-Bouldin Index']}\")\n",
    "    \n",
    "    # Store and display all best models\n",
    "    best_models = {\n",
    "        'Regression': best_regression_model,\n",
    "        'Classification': best_classification_model,\n",
    "        'Clustering': best_clustering_model\n",
    "    }\n",
    "    \n",
    "    return best_models\n",
    "\n",
    "# Run the comparison\n",
    "best_models = compare_models()\n",
    "print(\"\\n\\nMost Suitable Models:\")\n",
    "for i in best_models:\n",
    "    print(best_models[i])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa1ec1b9-1253-46b0-94f7-6d0cd41d8594",
   "metadata": {},
   "source": [
    "## Save Best Model (Random Forest Classifier)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "871bf718-b694-4e1a-a058-94cd055f65b8",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'best_model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[26], line 4\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mjoblib\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;66;03m# Assuming `best_model` is your trained model\u001b[39;00m\n\u001b[1;32m----> 4\u001b[0m joblib\u001b[38;5;241m.\u001b[39mdump(\u001b[43mbest_model\u001b[49m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mbest_model.pkl\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m      6\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mModel saved successfully!\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'best_model' is not defined"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "# Assuming `best_model` is your trained model\n",
    "joblib.dump(best_model, 'best_model.pkl')\n",
    "\n",
    "print(\"Model saved successfully!\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
