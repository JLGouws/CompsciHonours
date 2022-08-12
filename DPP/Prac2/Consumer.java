import org.jcsp.lang.*;

/** Consumer class: reads one int from input channel, displays it, then
  * terminates.
  */
public class Consumer implements CSProcess
  { private SharedChannelInput channel;
    private SharedChannelOutput reqChannel;
    private boolean itemsLeft;
    private final String name;
    private Integer[] items; 

    public Consumer (final SharedChannelOutput req, final SharedChannelInput in)
      { this(req, in, "Consumer");
      } // constructor

    public Consumer (final SharedChannelOutput req, final SharedChannelInput in, final String name)
      { channel    = in;
        reqChannel = req;
        this.name  = name;
        itemsLeft  = true;
        items = new Integer[100];
      } // constructor

    public Object[] getItems ()
      {
        return items;
      }
    public void run ()
      { 
        for (int i = 0; i <= 100 && itemsLeft; i++)
          {
            reqChannel.write(i == 100 ? null : -1);
            Object o = channel.read();
            if (o == null)
              itemsLeft = false;
            else
              items[i] = (Integer) o;
          }
      } // run

  } // class Consumer
