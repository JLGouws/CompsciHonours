import org.jcsp.lang.*;

/** Producer class: produces one random integer and sends on
  * output channel, then terminates.
  */
public class Producer implements CSProcess
  { private SharedChannelOutput channel;
    private final int s, r;

    public Producer (final SharedChannelOutput out)
      { this (out, 1, 100);
      } // constructor

    public Producer (final SharedChannelOutput out, final int a, final int b)
      { channel = out;
        r = (b - a + 1); //set range for this producer
        s = a;           //set offset for the producer production
      } // constructor

    public void run ()
      { int item;
        int num = 0, max = 0;
        for (int i = 0; i < 100; i++) //produce 100 items and send them on the channel to the buffer.
          { item = (int)(Math.random() * r) + s;
            channel.write(item);
          }
        channel.write(null);
      } // run

  } // class Producer
