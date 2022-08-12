/**
 * @author J L Gouws
 *
 * To ensure that the program terminates cleanly,
 * the buffer must know how many consumers and producers it is lauching in parallel with.
 * Consumers must consume enough items to ensure that all produced items are consumed.
 * The producer sends a null object when it has finished.
 * When all items have been produced the buffer closes down the communication
 * For this program the consumer can decide to end consumption by sending a null to the buffer.
 * This design allows each consumer to consume 100 items.
 */
import org.jcsp.lang.*;

/** Main program class for Producer/Consumer example.
  * Sets up channel, creates one of each process then
  * executes them in parallel, using JCSP.
  */
public final class testOne
  {
    public static void main (String[] args)
      { new testOne();
      } // main

    public testOne()
      { // Create channel object
        final Any2OneChannel pbChannel     = Channel.any2one(), //create channels for communication
                             bcReqChannel  = Channel.any2one();
        final One2AnyChannel bcChannel     = Channel.one2any();

        // Create and run parallel construct with a list of processes
        Consumer a = new Consumer(bcReqChannel.out(), bcChannel.in(), "a"), //create two consumers
                 b = new Consumer(bcReqChannel.out(), bcChannel.in(), "a"); //created here to get results later
        CSProcess[] procList = { 
            new Producer(pbChannel.out(), 1, 1000),                    //Create producers for the buffer
            new Buffer(pbChannel.in(), bcReqChannel.in(), bcChannel.out(), 1, 1), //launch buffer in parallel
            b 
        }; // Processes
        //launch processes in parallel
        Parallel par = new Parallel(procList); // PAR construct
        par.run(); // Execute processes in parallel
      } // PCMain constructor

  } // class PCMain
