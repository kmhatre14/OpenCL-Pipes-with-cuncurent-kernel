pipe unsigned char input_pipe __attribute__((xcl_reqd_pipe_depth(32)));
//pipe unsigned char output_pipe __attribute__((xcl_reqd_pipe_depth(401408)));
//kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void kernel0(__global unsigned char* input) {
    unsigned char i;
	int x=get_global_id(0);
    //printf("ip %d\t",input[x]);
    
    for (i = 0; i<100; i++)
    {
        input[i+x*100] = i;
        write_pipe_block(input_pipe, &input[i+x*100]);
        //printf("kernel 0 write %d \n",i);
    }
    printf("0\n");
}
//kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void kernel1(__global unsigned char* output) {
    
    //__local unsigned char localBuffer[12544];
	unsigned char output_data;
	int x=get_global_id(0);
    int i;

    //for (i = 0; i< 10 ; i++){
        //read_pipe_block(input_pipe, &output[x]);
    //}      
    //if ( x > 10 )
    //printf("%d\n",x);
    for (i = 0; i< 100 ; i++){
        read_pipe_block(input_pipe, &output[i+x*100]);
        //output[i] = 8;
        //printf("kernel 1 read %d \n",i);
    }
    //printf("op %d\t",output[x]);
    printf("1\n");
}
