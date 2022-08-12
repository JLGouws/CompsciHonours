import org.jcsp.lang.*;

/** Buffer class: receives an item from a Producer and sends it onto a Consumer 
  */
public class Buffer implements CSProcess
  { private AltingChannelInput inChannel,
                               outReqChannel;
    private ChannelOutput  outChannel;
    private final Alternative alt;
    private final int BUFF_LENGTH = 10;
    private final Object[] buff = new Object[10];
    private int numProducers,
                numConsumers;
    private static final int OUT  = 0,
                             IN   = 1;
    private int head,
                length;

    public Buffer (
                    final AltingChannelInput in,
                    final AltingChannelInput outReq,
                    final ChannelOutput out,
                    int numProducers,
                    int numConsumers
                  )
      {
        head   = 0;
        length = 0;
        inChannel = in;
        outReqChannel = outReq;
        outChannel = out;
        this.numProducers = numProducers;
        this.numConsumers = numConsumers;
        alt = new Alternative(
                new Guard[] 
                  { outReqChannel,
                    inChannel
                  }
              );
      } // constructor

    public void run ()
      { while(numConsumers > 0) 
          { 
           // if (length == BUFF_LENGTH)
           //   { outReqChannel.read();
           //     outChannel.write(buff[head]);
           //     head = (head + 1) % BUFF_LENGTH;
           //     length--;
           //   }
           // else if (length == 0)
           //   { buff[head] = inChannel.read();
           //     length++;
           //   }

            switch ((length % BUFF_LENGTH) != 0 ? alt.select() : 
                      (length == 10 || 
                        (numProducers == 0 && length == 0) ? 0 : 1)) 
              { case OUT:
                  Object o = outReqChannel.read();           //read from consumer
                  outChannel.write(buff[head]);
                  if (o == null || (numProducers == 0 && length == 0))
                      numConsumers--;
                  else
                    {
                      buff[head] = null;
                      head = (head + 1) % BUFF_LENGTH;
                      length--;
                    }
                  break;
                case IN:
                  Object Item = inChannel.read();
                  if (Item == null) //this producer is done
                    numProducers--;
                  else 
                    buff[(head + length++) % BUFF_LENGTH] = Item;
              }
          }
      } // run

  } // class Producer
