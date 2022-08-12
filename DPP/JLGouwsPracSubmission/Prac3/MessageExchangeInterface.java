import java.util.List;
import java.rmi.Remote;
import java.rmi.RemoteException;

public interface MessageExchangeInterface extends Remote
{
  /**
   * Store a message on the server
   */
  public void storeMessage(Message m) throws RemoteException;

  /**
   * Get the most recent message that matches the recipient of m
   */
  public Message accessNewestMessage(Message m) throws RemoteException;

  /**
   * Get all the messages that match the recipient of m
   */
  public List<Message> accessMessages(Message m) throws RemoteException;
}
