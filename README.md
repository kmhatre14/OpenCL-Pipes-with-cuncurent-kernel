# OpenCL-Pipes-with-cuncurent-kernel

The example given in this repository shows how to transfer data between 2 kernals without interacting with the host memory.
The depth of the pipe is 32 and the data transfer to be done is 1k. Thus for the pipe to work the data needs to be transfered
in chunks of 32. So both the kernels needs to run simultaniously or cuncurently to finish the job. The sync between the 2 kernels
is done by using the write_pipe_block() and read_pipe_block() command. 

kernel0 :- This is the input kernel this kernel is the producer that provides data to pipe

kernel1 :- This is the consumer kernel
