import org.jcsp.lang.*;

/** Producer class: produces one random integer and sends on
  * output channel, then terminates.
  */
public class Producer implements CSProcess
  { private SharedChannelOutputInt channel;
    private final int s, r;

    public Producer (final SharedChannelOutputInt out)
      { this (out, 1, 100);
      } // constructor

    public Producer (final SharedChannelOutputInt out, final int a, final int b)
      { channel = out;
        r = (b - a + 1);
        s = a;
      } // constructor

    public void run ()
      { int item;
        int num = 0, max = 0;
        for (int i = 0; i < 100; i++)
          { 
            /*
            try 
              { if (num == max);
                max = 5 + (int) (11 * Math.random());
                num = 0;
                Thread.sleep((int) (15 * Math.random()));
              }
            catch (Exception ex)
              {
              }
            num++;
            */
            item = (int)(Math.random() * r) + s;
            channel.write(item);
          }
      } // run

  } // class Producer
