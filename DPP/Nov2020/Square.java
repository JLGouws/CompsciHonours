import org.jcsp.lang.*;

public class Square implements CSProcess
{
  private AltingChannelInputInt inA;
  private AltingChannelInputInt inB;
  private ChannelOutputInt outC;
  private Alternative alt;

  public Square(AltingChannelInputInt inA, AltingChannelInputInt inB, ChannelOutputInt outC)
  {
    this.inA = inA;
    this.inB = inB;
    this.outC = outC;
    alt = new Alternative(
      new Guard[]
      {
        inA,
        inB
      }
    );
  }

  public void run()
  {
    while(true)
    {
      switch(alt.select())
      {
        case 0:
        {
          int x;
          x = inA.read();
          outC.write(x * x);
        }
          break;
        case 1:
        {
          int x;
          x = inB.read();
          outC.write(x * x);
        }
          break;
      }

    }
  }
}
