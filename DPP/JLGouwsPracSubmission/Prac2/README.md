See out.txt and out2.txt for the program's output

To ensure that the program terminates cleanly, the buffer must know how many consumers and producers it is lauching in parallel with.
Consumers must consume enough items to ensure that all produced items are consumed.
The producer sends a null object when it has finished.
When all items have been produced the buffer closes down the communication
For this program the consumer can decide to end consumption by sending a null to the buffer.
This design allows each consumer to consume 100 items.

The tasking mechanism of Java and JCSP seem fair.
Each consumer consumes roughly the same number of items from each producer.
One process seems to get scheduled for a little bit, and then another process get processing time.
In the output trace one producer seems dominant, and then production from another producer is more prevalent.

For the consumer output (out.txt), the consumers alternate their output.
This occurence is probably due to the JVM putting threads to sleep while the threads wait for the low IO.
