# 1. Start from an official, lightweight Python base image.
FROM python:3.11-slim

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Copy *only* the requirements file first.
# This is a cool Docker trick! It caches this layer, so if we
# only change our app code (main.py) and not our requirements,
# Docker doesn't need to reinstall all the packages every time.
COPY requirements.txt .

# 4. Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Now, copy the rest of our application code into the image.
COPY main.py .

# 6. Expose the port our Uvicorn server will run on.
EXPOSE 8000

# 7. Define the command to run when the container starts.
# This is the same command we used to run it locally.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]