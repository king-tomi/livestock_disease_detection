# Setting Up Python Virtual Environment and React Project

## Setting Up Python Virtual Environment

1. **Create a Virtual Environment:**

   ```bash
   python -m venv env
   ```

2. **Activate the Virtual Environment**:
    * On Windows

    ``` bash
    venv_name\Scripts\activate
    ```

    * On Mac/Linux

    ``` bash
    source venv_name/bin/activate
    ```

3. **Install Packages from requirements.txt:**

    ``` bash
    pip install -r requirements.txt
    ```

## Setting Up React Project from Existing Package

1. **Navigate to Your React Project Directory:**

    ``` bash
    cd alzheimer_project
    ```

2. **Install Dependencies:**

    ``` bash
    npm install
    ```

3. **Start the Development Server:**

    ``` bash
    npm start
    ```

4. **Access Your React App:**
You can now access your React app at <http://localhost:3000> in your browser.

## PLS NOTE

The code below is the API query part

``` jsx
        const response = await axios.post("/upload/", formData, {
            headers: {
            "Content-Type": "multipart/form-data",
            },
        });
```

Replace the `upload` with the endpoint of your API
