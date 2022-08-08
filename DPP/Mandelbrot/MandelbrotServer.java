import java.rmi.registry.Registry;
import java.rmi.registry.LocateRegistry;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class MandelbrotServer implements MandelbrotCalculator
  {     public byte[][] calculateMandelbrot (int xsize, int ysize, double x1, double x2, double y1, double y2)
      { double x, y, xx, a, b = y1, da = x2/xsize, db = y2/ysize;
        byte[][] points = new byte[xsize][ysize];

        for (int i = 0; i < ysize; i++)
          { a = x1;
            for (int j = 0; j < xsize; j++)
              { byte n = 0;
                x = 0.0;
                y = 0.0;
                while ( (n < 100) && ( (x*x)+(y*y) < 4.0) )
                  { xx = x * x - y * y + a;
                    y = 2 * x * y + b;
                    x = xx;
                    n++;
                  }
                points[j][i] = n;
                a = a + da;
              }
            b = b + db;
          }
        return points;
      } // calculateMandelbrot

    public static void main (String args[])
      { try
          { MandelbrotServer server = new MandelbrotServer();
      	    MandelbrotCalculator stub = (MandelbrotCalculator)UnicastRemoteObject.exportObject(server, 0);

      	    // Bind the remote object's stub in the registry
      	    Registry registry = LocateRegistry.getRegistry();
      	    registry.bind("MandelbrotService", stub);

      	    System.err.println("Mandelbrot Server ready");
          }
        catch (Exception e)
          { System.err.println("Server exception: " + e.toString());
	          e.printStackTrace();
          }

      } // main

  } // class MandelbrotServer
