/** Simple generic queue class, using an array.
  * @author George Wells
  * @version 2.0 (3 January 2005)
  */
public class ArrayQueue<T>
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
    public ArrayQueue ()
      { data = (T[])new Object[10];
        hd = tl = -1;
      } // Constructor

    /** Add an item to the back of a queue.
      * <BR><I>Precondition:</I> There is space available in the queue.
      * <BR><I>Postcondition:</I> The queue is not empty.
      * @param item The item to be added to the queue.
      */
    public void add (T item)
      { tl = (tl + 1) % data.length;
        assert tl != hd : "Space is available";
        data[tl] = item;
        if (hd == -1) // First item in queue
          hd = tl;
      } // add

    /** Remove an item from the front of a queue.
      * <BR><I>Precondition:</I> The queue is not empty.
      * @return The item removed from the front of the queue.
      */
    public T remove ()
      { assert hd != -1 : "Queue is not empty";
        int tmpIndex = hd;
        if (hd == tl) // Was last element
          hd = tl = -1;
        else
          hd = (hd + 1) % data.length;
        return data[tmpIndex];
      } // remove

  } // class ArrayQueue
