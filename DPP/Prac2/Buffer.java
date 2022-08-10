import org.jcsp.lang.*;

/** Buffer class: receives an item from a Producer and sends it onto a Consumer 
  */
public class Buffer implements CSProcess
  { private AltingChannelInputInt inChannel,
                                  outReqChannel;
    private ChannelOutputInt outChannel;
    private final Alternative alt;
    private final int BUFF_LENGTH = 10;
    private final int[] buff = new int[10];
    private static final int OUT =  0,
                             IN  = 1; 
    private int head,
                length;

    public Buffer (
                    final AltingChannelInputInt in,
                    final AltingChannelInputInt outReq,
                    final ChannelOutputInt out
                  )
      {
        head   = 0;
        length = 0;
        inChannel = in;
        outReqChannel = outReq;
        outChannel = out;
        alt = new Alternative(
                new Guard[] 
                  { outReqChannel,
                    inChannel
                  }
              );
      } // constructor

    public void run ()
      { while(true) 
          { if (length == BUFF_LENGTH)
              { outReqChannel.read();
                outChannel.write(buff[head]);
                head = (head + 1) % BUFF_LENGTH;
                length--;
              }
            else if (length == 0)
              { buff[head] = inChannel.read();
                length++;
              }

            switch (alt.select()) 
              { case OUT:
                  outReqChannel.read();           //read from consumer
                  outChannel.write(buff[head]);
                  head = (head + 1) % BUFF_LENGTH;
                  length--;
                  break;
                case IN:
                  buff[(head + length) % BUFF_LENGTH] = inChannel.read();
                  length++;
                  break;
              }
          }
      } // run

  } // class Producer
