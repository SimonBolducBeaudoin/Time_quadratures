# Time_quadratures
TBR    
    
# Building and compiling
Edit config.cmake for your machine (If you are compiling in a different envionnment than your python installation) so th  at pybind11 can be detected and used.
```bash
cmake -S . -B ./build  # Building in Linux
```
```bash
cmake -S . -B ./build -DCMAKE_TOOLCHAIN_FILE=../CMakeConfigs/mingw_toolchain.cmake # Building in Windows (Cross compiling on Cygwin with mingw)
```
```bash
cmake --build build/ # Compiles the project
```
```bash
cmake --install build/ # Copies files to their intended directory
```   
```bash
cmake --build build/ --target uninstall # uninstall 
```  
```bash
cmake --build build/ --target clean # removes targets 
```   
```bash
rm -R -f build/ # removes build directory
```   