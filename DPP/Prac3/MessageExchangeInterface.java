import java.rmi.Remote;
import java.rmi.RemoteException;

public interface MessageExchangeInterface extends Remote
{
  public void storeMessage(Message m) throws RemoteException;

  public Message accessNewestMessage(Message m) throws RemoteException;
}
