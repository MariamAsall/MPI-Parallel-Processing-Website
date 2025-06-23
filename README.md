# MPI-Parallel-Processing-Website
## MPI Parallel Data Processing Demo

This project is a web-based demo for parallel data processing using MPI (Message Passing Interface) and Python. It allows users to run various data processing tasks (like sorting, word count, image filtering, linear regression, keyword search, statistics, and matrix multiplication) in parallel, comparing performance to sequential execution. The user interface is built with Streamlit, and the backend leverages mpi4py for MPI operations.

---

## Features

- **Multiple Tasks:** Sorting, Word Count, Image Filtering, Linear Regression, Keyword Search, Statistics, Matrix Multiplication.
- **Parallel Execution:** Choose the number of MPI processes (2–32) to see the speedup over sequential processing.
- **User-Friendly Web UI:** Built with Streamlit for easy interaction.
- **Performance Comparison:** Displays execution time and speedup between sequential and parallel runs.

---

## Step-by-Step Usage and Explanation

### 1. Installation & Setup

**Requirements:**
- Python 3.x
- mpi4py
- Streamlit
- NumPy, pandas, scikit-learn, Pillow, OpenCV

**Install dependencies:**
```bash
pip install mpi4py streamlit numpy pandas scikit-learn pillow opencv-python
```

**MPI Installation:**  
You must have an MPI implementation (like OpenMPI or MPICH) installed on your system.

---

### 2. Running the Application

**Start the Streamlit app:**
```bash
streamlit run mpi-web-app.py
```
This launches the web interface in your browser.

![image](https://github.com/user-attachments/assets/64353f2e-0236-43e0-bd80-8a282b74e05e)


### 3. Using the Web Interface

#### **a. Select a Task**

- Use the sidebar to choose a task:
  - Sorting
  - File Word-Count
  - Image Filter
  - Linear Regression
  - Keyword Search
  - Statistics
  - Matrix Multiplication
![image](https://github.com/user-attachments/assets/69181134-d068-4335-bb4c-e919272be80f)


#### **b. Set MPI Processes**

- Adjust the slider to select the number of MPI processes (from 2 to 32).
  
![image](https://github.com/user-attachments/assets/8c0b3b6d-df81-4f56-a060-90c5bd4d84cd)


#### **c. Input Data**

- Depending on the task, provide the required input:
  - **Sorting:** Enter comma-separated numbers.
  - **File Word-Count:** Upload a `.txt` file.
  - **Image Filter:** Upload an image and select a filter (Grayscale or Blur).
  - **Linear Regression:** Upload a CSV file with features and target column.
  - **Keyword Search:** Upload a large text file and enter a keyword.
  - **Statistics:** Upload a CSV file with numeric data.
  - **Matrix Multiplication:** Upload two CSV files representing matrices A and B.

#### **d. Run the Task**

- Click the **Run** button to execute the task.
- The app will run both sequential and parallel versions, display the results, and show a performance comparison (including speedup).

---

## How It Works (Backend Explanation)

### **A. Sequential vs. Parallel Execution**

- For each task, there are two implementations:
  - **Sequential:** Runs in a single process, used as a baseline.
  - **Parallel (MPI):** The data is split among multiple processes using mpi4py, and the results are combined at the end[1].

### **B. Task Execution Flow**

1. **User Input:** Data and parameters are collected from the UI.
2. **Sequential Run:** The task is executed in the main process for benchmarking.
3. **Parallel Run:** The app spawns an MPI job using `mpiexec` with the selected number of processes. The actual computation is distributed across processes, and results are gathered on the root process.
4. **Results & Timing:** Both results and timings are displayed, allowing users to compare performance.

### **C. Example: Sorting Task**

- Numbers are split among MPI processes.
- Each process sorts its chunk and participates in an odd-even transposition sort for global ordering.
- Results are gathered and displayed, along with the elapsed time for both sequential and parallel runs[1].

---

## Tasks Supported

| Task                | Input Type         | Output Example         | Parallelization Approach   |
|---------------------|-------------------|------------------------|---------------------------|
| Sorting             | Numbers           | Sorted list            | Distributed sort          |
| File Word-Count     | Text file         | Word counts            | Chunked by lines          |
| Image Filter        | Image file        | Filtered image         | Split by image rows       |
| Linear Regression   | CSV               | Coefficients           | Training on root          |
| Keyword Search      | Text file, string | Occurrences, positions | Chunked by text segment   |
| Statistics          | CSV               | Stats per column       | Split by rows             |
| Matrix Multiplication | Two CSVs        | Result matrix          | Block-wise multiplication |

---

## Notes

- **Performance:** Speedup depends on the size of the data and the number of MPI processes. For small datasets, parallel overhead may outweigh benefits.
- **MPI Environment:** The app must be run in an environment where `mpiexec` is available and properly configured.
- **Platform:** Tested on Linux; some MPI implementations may behave differently on Windows.

---

## Extending the Project

- Add new data processing tasks by implementing both sequential and MPI versions.
- Improve front-end with more visualizations or input validation.
- Add support for larger datasets and streaming data.

---

## Troubleshooting

- If you encounter errors related to MPI, ensure your MPI installation is working (try `mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"`).
- For Python dependency issues, check your environment and reinstall required packages.

---

## License

This project is provided for educational and demonstration purposes.

---

## Contact

For questions or contributions, please open an issue or submit a pull request.

---

**References:**  
- The implementation details and task logic are based on the provided `mpi-web-app.py` file[1].  
- UI screenshots and workflow are based on the attached images.
