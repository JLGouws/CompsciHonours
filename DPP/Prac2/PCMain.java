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
 *
 * The tasking mechanism of Java and JCSP seem fair.
 * Each consumer consumes roughly the same number of items from each producer.
 * One process seems to get scheduled for a little bit, and then another process get processing time.
 * In the output trace one producer seems dominant, and then production from another producer is more prevalent.
 */
import org.jcsp.lang.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;

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
      { int prod1 = 0, prod2 = 0;
        for (int i = 0; i < os.length; i++)
          {
            System.out.println("Item " + (i + 1) + " is " + os[i]);
            if ((Integer) os[i] >= 1001)
              prod2++;
            else
              prod1++;

          }
          System.out.println("Consumed " + prod1 + " items from producer1");
          System.out.println("Consumed " + prod2 + " items from producer2");
      }

    public PCMain ()
      { // Create channel object
        PrintStream console = System.out;
        File file = null; 
        FileOutputStream fos = null;
        PrintStream ps = null;
        //switch output to another file so that the ouput is easier to read
        try
          {
            file = new File("out.txt");
            fos  = new FileOutputStream(file);
            ps   = new PrintStream(fos);
          }
        catch (Exception e)
          {
            System.err.println("Couldn't open file stream" + e);
          }
        System.setOut(ps);
        final Any2OneChannel pbChannel     = Channel.any2one(), //create channels for communication
                             bcReqChannel  = Channel.any2one();
        final One2AnyChannel bcChannel     = Channel.one2any();

        // Create and run parallel construct with a list of processes
        Consumer a = new Consumer(bcReqChannel.out(), bcChannel.in()), //create two consumers
                 b = new Consumer(bcReqChannel.out(), bcChannel.in()); //created here to get results later
        CSProcess[] procList = { 
            new Producer(pbChannel.out(), 1, 1000),                    //Create producers for the buffer
            new Producer(pbChannel.out(), 1001, 2000), 
            new Buffer(pbChannel.in(), bcReqChannel.in(), bcChannel.out(), 2, 2), //launch buffer in parallel
            a,
            b 
        }; // Processes
        //launch processes in parallel
        Parallel par = new Parallel(procList); // PAR construct
        par.run(); // Execute processes in parallel
        Object[] as = a.getItems(), //get the results of execution
                 bs = b.getItems();
        System.out.println("Consumer A consumed: "); //print the results for each consumer
        printArray(as);
        System.out.println("Consumer B consumed: ");
        printArray(bs);
        System.setOut(console);
        System.out.println("Output written to out.txt");
      } // PCMain constructor

  } // class PCMain
