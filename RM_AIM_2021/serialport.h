#ifndef _SERIALPORT_H_
#define _SERIALPORT_H_
#include<iostream>
#include<stdio.h>
#include<string.h>
#include<sys/types.h>
#include<errno.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<termios.h>
#include<stdlib.h>
#include<sys/ioctl.h>

using namespace std;

//@brief:
//@brief:linux下的串口通信类，可以通过构造函数直接打开一个串口，并初始化（默认9600波特率，8位数据，无奇偶校验，1位停止位）
//             send()成员函数可以直接发送字符串，set_opt()更改参数。串口会在析构函数中自动关闭
//@example:Serialport exp("/dev/ttyUSB0");
//                  exp.set_opt(115200,8,'N',1);
//
class Serialport
{
public:
    int fd ;
    char tmpchar[12];
    const char *buffer;
    unsigned char rData[255];

public:
    Serialport(string port);//定义Serialport类的成员函数，
    Serialport();
    ~ Serialport();
    int open_port(string port);
    int set_opt(int nSpeed = 9600 , int nBits = 8, char nEvent ='N', int nStop = 1);
    bool send(char *str);
    bool sendAngle(float angleYaw,float anglePitch);
    bool sendXYZ(double * xyz);
    bool sendXY(float * xy);
    void readMode(int &carMode);
};

//@brief:linux下的串口通信类的成员函数。
//       open_port()成员函数可以打开一个串口，set_opt()更改参数。
//@example:open_port("/dev/ttyUSB0");
//         set_opt(115200, 8, 'N', 1);
Serialport::Serialport(string port)
{
        open_port(port);
        set_opt();
}

//Serialport下的成员函数open_port()的实现；
int Serialport::open_port(string port)
{
    // char *dev[]={"/dev/ttyS0","/dev/ttyS1","/dev/ttyS2"};
    // fd = open( "/dev/ttyS0", O_RDWR|O_NOCTTY|O_NDELAY);
    // fd = open( "/dev/ttyUSB0", O_RDWR|O_NOCTTY|O_NDELAY);
    fd = open(port.c_str() , O_RDWR|O_NOCTTY|O_NDELAY);
    if (-1 == fd)
    {
        printf("\n\e[31m\e[1m ERROR:打开串口%s失败 \e[0m\n\n",port.c_str());
        return -1;
    }
    else
    {
        fcntl(fd, F_SETFL, 0);
    }

    if(fcntl(fd, F_SETFL, 0) < 0)
        printf("fcntl failed!\n");
    else
        printf("fcntl=%d\n",fcntl(fd, F_SETFL,0));

    if(isatty(STDIN_FILENO)==0)
        printf("standard input is not a terminal device\n");
    else
        printf("isatty success!\n");

    //printf("fd-open=%d\n",fd);

    return fd;
}

/*设置串口属性：
fd: 文件描述符
nSpeed: 波特率
nBits: 数据位
nEvent: 奇偶校验
nStop: 停止位*/

int Serialport::set_opt(int nSpeed , int nBits, char nEvent , int nStop )
{
    struct termios newtio,oldtio;
    if ( tcgetattr( fd,&oldtio) != 0)
    {
        perror("SetupSerial error");
        return -1;
    }
    bzero( &newtio, sizeof( newtio ) );
    newtio.c_cflag |= CLOCAL | CREAD;
    newtio.c_cflag &= ~CSIZE;
    switch( nBits )
    {
    case 7:
        newtio.c_cflag |= CS7;
        break;
    case 8:
        newtio.c_cflag |= CS8;
        break;
    //case 9:
    //newtio.c_cflag |= CS9;
    //break;
    }
    switch( nEvent )
    {
    case 'O':
        newtio.c_cflag |= PARENB;
        newtio.c_cflag |= PARODD;
        newtio.c_iflag |= (INPCK | ISTRIP);
        break;
    case 'E':
        newtio.c_iflag |= (INPCK | ISTRIP);
        newtio.c_cflag |= PARENB;
        newtio.c_cflag &= ~PARODD;
        break;
    case 'N':
        newtio.c_cflag &= ~PARENB;
        break;
    }
    switch( nSpeed )
    {
    case 2400:
        cfsetispeed(&newtio, B2400);
        cfsetospeed(&newtio, B2400);
        break;
    case 4800:
        cfsetispeed(&newtio, B4800);
        cfsetospeed(&newtio, B4800);
        break;
    case 9600:
        cfsetispeed(&newtio, B9600);
        cfsetospeed(&newtio, B9600);
        break;
    case 115200:
        cfsetispeed(&newtio, B115200);
        cfsetospeed(&newtio, B115200);
        break;
    default:
        cfsetispeed(&newtio, B9600);
        cfsetospeed(&newtio, B9600);
        break;
    }

    if( nStop == 1 )
        newtio.c_cflag &= ~CSTOPB;
    else if ( nStop == 2 )
        newtio.c_cflag |= CSTOPB;

    newtio.c_cc[VTIME] = 0;
    newtio.c_cc[VMIN] = 0;
    tcflush(fd,TCIFLUSH);

    if((tcsetattr(fd,TCSANOW,&newtio))!=0)
    {
        perror("com set error");
        return -1;
    }
    printf("Serial port set done!\n");
    return 0;
}

bool Serialport::send(char *str)
{
    buffer = str;
    if(fd < 0 || write(fd, buffer, 1) < 0)
    {
        perror("\n\e[31m\e[1m ERROR:串口通信失败 \e[0m\n\n");

        return false;
    }

    return true;
}

bool Serialport::sendAngle(float _angle1,float _angle2)
{
    char send_num=0;
    unsigned char *p;
    memset(tmpchar, 0x00, sizeof(tmpchar));    //对tempchar清零
    tmpchar[0] = 0xA5;                                        //起始标志
    tmpchar[1] = 0x5A;
    //tmpchar[2] = 0x06;                                //the number of data bytes
    p=(unsigned char *)&_angle1;
    tmpchar[2] = *p;                      //第一个角度的低8位
    tmpchar[3] = *(p+1);                  //第一个角度
    tmpchar[4] = *(p+2);                  //
    tmpchar[5] = *(p+3);                  //
    p=(unsigned char *)&_angle2;
    tmpchar[6] = *p;                      //第二个角度的低8位
    tmpchar[7] = *(p+1);                  //第二个角度
    tmpchar[8] = *(p+2);                  //
    tmpchar[9] = *(p+3);                  //
    tmpchar[10] = 0xAA;                   //Check
    tmpchar[11] = 0xAA;                   //End
    //tmpchar[7] = 0xFE;                  //结束标志
    cout << "Send_Angle:"<<dec<<_angle1<<","<<dec<<_angle2<<endl;
    cout << "Send_data:" <<hex<<(int)tmpchar[2]<<","<<hex<<(int)tmpchar[3]<<","<<hex<<(int)tmpchar[4]<<","<<hex<<(int)tmpchar[5]<<endl;
    cout << "Send_data:" <<hex<<(int)tmpchar[6]<<","<<hex<<(int)tmpchar[7]<<","<<hex<<(int)tmpchar[8]<<","<<hex<<(int)tmpchar[9]<<endl;
    cout << "Check:" << hex << (int)tmpchar[10] << endl;
    for( send_num = 0; send_num < 12; send_num++)
    {
        if(!send(tmpchar + send_num))
            return false;
    }
    cout<<"Send successfully!"<<endl;
    return true;
}

bool Serialport::sendXYZ(double * xyz)
{
    unsigned char send_bytes[]={0xFF,0x00,0x00,0x00,0x00,0x00,0x00,0xFE};
    if(NULL==xyz)
    {
        if(8==write(fd,send_bytes,8))
            return true;
        return false;
    }
    short *data_ptr=(short *)(send_bytes+1);
    data_ptr[0]=(short)xyz[0];
    data_ptr[1]=(short)xyz[1];
    data_ptr[2]=(short)xyz[2];
    if(8==write(fd,send_bytes,8))
        return true;
    return false;
}
bool Serialport::sendXY(float * xy)
{
    unsigned char send_bytes[]={0xFF,0x00,0x00,0x00,0x00,0xFE};
    if(NULL==xy)
    {
        if(6==write(fd,send_bytes,6))
            return true;
        return false;
    }
    short *data_ptr=(short *)(send_bytes+1);
    data_ptr[0]=(short)xy[0];
    data_ptr[1]=(short)xy[1];
    //printf("%d\n",send_bytes[5]);
    if(6==write(fd,send_bytes,6))
        return true;
    return false;
}

// 0xA5 | 0x5A | hitmode | 0xff
void Serialport::readMode(int &carMode)
{
    int bytes;
    ioctl(fd, FIONREAD, &bytes);
    if(bytes == 0) return;
    //cout<<"get read mode!"<<endl;
    bytes = read(fd,rData,6);
    printf("receive0:%d\r\n",rData[0]);
    printf("receive1:%d\r\n",rData[1]);
    printf("receive2:%d\r\n",rData[2]);
    printf("receive3:%d\r\n",rData[3]);
    if(rData[0] == 0xA5)
    {
        carMode = rData[2];
        printf("receive mode:%d\r\n",carMode);
    }
    /*ioctl(fd, FIONREAD, &bytes);
    if(bytes>0)
    {
        read(fd,rData,bytes);
    }
    cout << "readMode end!!!\n" << endl;*/
}

Serialport:: ~Serialport()
{
    close(fd);
}

#endif
