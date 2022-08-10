/** Simple generic queue class, using an array, with thread
  * synchronisation.
  * @author George Wells
  * @version 2.1 (26 August 2008)
  */
public class ArrayQueueSync<T>
  { /** The array of data.
      */
    private T[] data;
    /** Index of the item at the head (front) of the queue.
      */
    private int hd;
    /** Index of the item at the tail (back) of the queue.
      */
    private int tl;

    /** Create an empty queue, with a capacity for 10 items.
      * <BR><I>Precondition:</I> none.
      * <BR><I>Postcondition:</I> The queue is initialised and is empty.
      * @param initSize The maximum size of the queue.
      */
    @SuppressWarnings("unchecked")
    public ArrayQueueSync ()
      { data = (T[])new Object[10];
        hd = tl = -1;
      } // Constructor

    /** Add an item to the back of a queue.
      * <BR><I>Precondition:</I> There is space available in the queue.
      * <BR><I>Postcondition:</I> The queue is not empty.
      * @param item The item to be added to the queue.
      */
    public synchronized void add (T item)
      { while (((tl + 1) % data.length) == hd) // No space
          { try
              { wait();
              }
            catch (InterruptedException exc)
              { // ignore
              }
          } // while
        tl = (tl + 1) % data.length;
        data[tl] = item;
        if (hd == -1) // First item in queue
          hd = tl;
        notifyAll(); // Release any waiting consumers
      } // add

    /** Remove an item from the front of a queue.
      * <BR><I>Precondition:</I> The queue is not empty.
      * @return The item removed from the front of the queue.
      */
    public synchronized T remove ()
      { while (hd == -1) // No data
          { try
              { wait();
              }
            catch (InterruptedException exc)
              { // ignore
              }
          } // while
        int tmpIndex = hd;
        if (hd == tl) // Was last element
          hd = tl = -1;
        else
          hd = (hd + 1) % data.length;
        notifyAll(); // Release any waiting producers
        return data[tmpIndex];
      } // remove

  } // class ArrayQueueSync
