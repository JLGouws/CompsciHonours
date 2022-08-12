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
        head   = 0; //set the head and length of the queue
        length = 0;
        inChannel = in;         // for a producer to send times to the buffer
        outReqChannel = outReq; // for the consumer to request an item from the buffer
        outChannel = out;       // for the buffer to send an item to a waiting consumer
        this.numProducers = numProducers;
        this.numConsumers = numConsumers;
        alt = new Alternative( //alternative for producer to request or producer the send
                new Guard[] 
                  { outReqChannel,
                    inChannel
                  }
              );
      } // constructor

    public void run ()
      { while(numConsumers > 0) //carry on while there are still consumers.
          { 
            switch ((length % BUFF_LENGTH) != 0 ? alt.select() :      //multiplex this
                      (length == 10 || 
                        (numProducers == 0 && length == 0) ? 0 : 1)) 
              { case OUT:                                             //need to send message to consumer
                  Object o = outReqChannel.read();                    //read from consumer
                  if (o == null || (numProducers == 0 && length == 0))//if this consumer wants to end or
                    { numConsumers--;                                 //there are no more items coming or this consumer is done
                      outChannel.write(null);                         //we end this consumer
                    }
                  else
                    {
                      outChannel.write(buff[head]);                   //write head onto the channel
                      head = (head + 1) % BUFF_LENGTH;
                      length--;
                    }
                  break;
                case IN:                                              //need to read an item from a producer
                  Object Item = inChannel.read();
                  if (Item == null)                                   //this producer is done
                    numProducers--;
                  else 
                    buff[(head + length++) % BUFF_LENGTH] = Item;     //add item to buffer
              }
          }
      } // run

  } // class Producer
