import org.jcsp.lang.*;

/** Consumer class: reads one int from input channel, displays it, then
  * terminates.
  */
public class Consumer implements CSProcess
  { private SharedChannelInput channel;
    private SharedChannelOutput reqChannel;
    private boolean itemsLeft;
    private Integer[] items; 
    private String name; 

    public Consumer (final SharedChannelOutput req, final SharedChannelInput in, String name)
      { channel    = in;
        reqChannel = req;
        itemsLeft  = true; //are there items left to consume
        items      = new Integer[100];
        this.name  = name;
      } // constructor

    public Object[] getItems ()
      { return items; //get the items that this consumer consumed
      }

    public void run ()
      { for (int i = 0; i <= 100 && itemsLeft; i++) // consume 100 items or until there are no items left
          { reqChannel.write(i == 100 ? null : -1); // request an item from the buffer
            Object o = channel.read();              // get buffer item
            if (o == null)
              itemsLeft = false;                    // The buffer told us to end
            else
              items[i] = (Integer) o;               //This is a legitimate itme store it
          }
        for (int i = 0; i < 100; i++)
          System.out.println(name + " consumed " + items[i] + " as item " + (i + 1));
      } // run

  } // class Consumer
