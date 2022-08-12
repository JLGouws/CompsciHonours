import org.jcsp.lang.*;

/** Main program class for Producer/Consumer example.
  * Sets up channel, creates one of each process then
  * executes them in parallel, using JCSP.
  */
public final class PCMain
  {
    public static void main (String[] args)
      { new PCMain();
      } // main

    private void printArray(Object[] os)
      {
        for (int i = 0; i < os.length; i++)
        {
          System.out.println("Item " + i + " is " + os[i]);
        }
      }

    public PCMain ()
      { // Create channel object
        final Any2OneChannel pbChannel     = Channel.any2one(),
                             bcReqChannel  = Channel.any2one();
        final One2AnyChannel bcChannel     = Channel.one2any();

        // Create and run parallel construct with a list of processes
        Consumer a = new Consumer(bcReqChannel.out(), bcChannel.in(), "Consumer a"),
                 b =new Consumer(bcReqChannel.out(), bcChannel.in(), "Consumer b");
        CSProcess[] procList = { 
            new Producer(pbChannel.out(), 1, 1000),
            new Producer(pbChannel.out(), 1001, 2000), 
            new Buffer(pbChannel.in(), bcReqChannel.in(), bcChannel.out(), 2, 2),
            a,
            b 
        }; // Processes
        Parallel par = new Parallel(procList); // PAR construct
        par.run(); // Execute processes in parallel
        Object[] as = a.getItems(),
                 bs = b.getItems();
        System.out.println("Consumer A consumed: ");
        printArray(as);
        System.out.println("Consumer B consumed: ");
        printArray(bs);
      } // PCMain constructor

  } // class PCMain
