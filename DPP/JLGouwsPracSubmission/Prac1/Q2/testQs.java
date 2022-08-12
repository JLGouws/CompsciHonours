/**
 * @author J L Gouws
 *
 * Parallel Quick Sort took an average of: 179,30
 * Sequential Quick Sort took an average of: 893,20
 * This result indicates a speed up of:
 * T_sequential/ T_parallel = 4.982
 */

import java.util.Random;

public class testQs 
  {
    public static void main(String[] args)
        { long startTime, endTime;
          double pAverage = 0,
                 sAverage = 0;
          Random randGen = new Random();
          Integer[] numbers  = new Integer[1000000],
                    numbers2 = new Integer[1000000];
          for (int j = 0; j < numbers.length; j++)//generate two lists of random numbers for testing
            numbers[j] = numbers2[j] = randGen.nextInt(1000);

          //run sorts to make sure the code has been jitted for accurate testing
          ParallelSort.<Integer>array(numbers);

          Sort.<Integer>array(numbers2);

          for (int i = 0; i < 10; i++)
            { for (int j = 0; j < numbers.length; j++)
                numbers[j] = numbers2[j] = randGen.nextInt(1000);//generate a new list of random numbers
              startTime = System.currentTimeMillis();
              ParallelSort.<Integer>array(numbers);//sort the array in parallel
              endTime = System.currentTimeMillis();

              pAverage += (endTime - startTime) / 10.;

              startTime = System.currentTimeMillis();
              Sort.<Integer>array(numbers2);//sequentially sort the array.
              endTime = System.currentTimeMillis();
              
              sAverage += (endTime - startTime) / 10.;
            }


          System.out.printf("Parallel Quick Sort took an average of: %.2f\n", pAverage);

          System.out.printf("Sequential Quick Sort took an average of: %.2f\n", sAverage);

        }
  }
