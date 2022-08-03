public class ProducerConsumer
  { private static final int NUM_ITERS = 100;
  
    public static void main (String[] args)
      { // Create queue
        ArrayQueue<Integer> buffer = new ArrayQueue<Integer>();
        // Create threads
        Producer p = new Producer(buffer);
        Consumer c = new Consumer(buffer);
        Thread t1 = new Thread(p);
        t1.setName("Producer");
        Thread t2 = new Thread(c);
        t2.setName("Consumer");
        System.out.println("Starting threads...");
        t1.start();
        t2.start();
        try
          { t1.join();
            t2.join();
          }
        catch (InterruptedException exc)
          { System.err.println("Interrupted while waiting for threads");
          }
        System.out.println("Threads finished.");
      } // main
      
    // Inner classes
    
    private static class Producer implements Runnable
      { private ArrayQueue<Integer> buffer;

        public Producer (ArrayQueue<Integer> buffer)
          { this.buffer = buffer;
          } // constructor

        public void run ()
          { for (int k = 0; k < NUM_ITERS; k++)
              { int rnd = (int)(Math.random() * 100);
                buffer.add(rnd);
                System.out.println("Producer: put " + rnd);
              } // for
          } // run
      } // inner class Producer
    
    private static class Consumer implements Runnable
      { private ArrayQueue<Integer> buffer;

        public Consumer (ArrayQueue<Integer> buffer)
          { this.buffer = buffer;
          } // constructor

        public void run ()
          { for (int k = 0; k < NUM_ITERS; k++)
              { int num = buffer.remove();
                System.out.println("Consumer: got " + num + "  Squared = " + (num*num));
              } // for
          } // run
      } // inner class Consumer

  } // class ProducerConsumer
