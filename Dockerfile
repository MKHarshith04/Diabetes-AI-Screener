# 1. Use the official Julia cloud environment
FROM julia:1.10

# 2. Create a folder for our app
WORKDIR /app

# 3. Copy our files into the cloud server
COPY Project.toml Manifest.toml ./
COPY app.jl Feature_Importance.png ./

# 4. Install all the required Julia packages
RUN julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

# 5. Tell the server to run the app
CMD ["julia", "--project=.", "app.jl"]
