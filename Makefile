CC = nvcc
CFLAGS = -std=c++11 -Wno-deprecated-gpu-targets `pkg-config opencv --cflags --libs`
INCLUDES = 
DFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
SOURCES = histogram_parallel.cu
OUTF = histogram_cuda.exe
OBJS = histogram_parallel.o

$(OUTF): $(OBJS)
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	$(CC) $(CFLAGS) -c $< 

rebuild: clean $(OUTF)

clean:
	rm *.o $(OUTF)

