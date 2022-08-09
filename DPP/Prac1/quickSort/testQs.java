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
          for (int i = 0; i < 10; i++)
            { for (int j = 0; j < numbers.length; j++)
                numbers[j] = numbers2[j] = randGen.nextInt(1000);
              startTime = System.currentTimeMillis();
              ParallelSort.<Integer>array(numbers);
              endTime = System.currentTimeMillis();

              pAverage += (endTime - startTime) / 10.;

              startTime = System.currentTimeMillis();
              Sort.<Integer>array(numbers2);
              endTime = System.currentTimeMillis();
              
              sAverage += (endTime - startTime) / 10.;
            }


          System.out.printf("Parallel Quick Sort took an average of: %.2f\n", pAverage);

          System.out.printf("Sequential Quick Sort took an average of: %.2f\n", sAverage);

        }
  }
