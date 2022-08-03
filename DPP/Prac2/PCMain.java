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

    public PCMain ()
      { // Create channel object
        final Any2OneChannelInt pbChannel = Channel.any2oneInt();
        final One2AnyChannelInt bcChannel = Channel.one2anyInt();
        final Any2OneChannelInt bcReqChannel = Channel.any2oneInt();

        // Create and run parallel construct with a list of processes
        CSProcess[] procList = { 
            new Producer(pbChannel.out(), 1, 1000), 
            new Producer(pbChannel.out(), 1001, 2000), 
            new Buffer(pbChannel.in(), bcReqChannel.in(), bcChannel.out()), 
            new Consumer(bcReqChannel.out(), bcChannel.in(), "Consumer a"),
            new Consumer(bcReqChannel.out(), bcChannel.in(), "Consumer b") 
        }; // Processes
        Parallel par = new Parallel(procList); // PAR construct
        par.run(); // Execute processes in parallel
      } // PCMain constructor

  } // class PCMain
