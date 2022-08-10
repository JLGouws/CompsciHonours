import org.jcsp.lang.*;

/** Consumer class: reads one int from input channel, displays it, then
  * terminates.
  */
public class Consumer implements CSProcess
  { private SharedChannelInputInt channel;
    private SharedChannelOutputInt reqChannel;
    private final String name;

    public Consumer (final SharedChannelOutputInt req, final SharedChannelInputInt in)
      { this(req, in, "Consumer");
      } // constructor

    public Consumer (final SharedChannelOutputInt req, final SharedChannelInputInt in, final String name)
      { channel    = in;
        reqChannel = req;
        this.name = name;
      } // constructor

    public void run ()
      { int[] item = new int[100];
        
        for (int i = 0; i < 100; i++)
          {
            reqChannel.write(-1);
            item[i] = channel.read();
          }
        for (int i = 0; i < 100; i++)
          System.out.println(name + " has consumed: " + item[i]);
      } // run

  } // class Consumer
