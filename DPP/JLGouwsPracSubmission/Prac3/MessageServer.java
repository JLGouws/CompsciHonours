import java.util.List;
import java.util.Hashtable;
import java.util.LinkedList;

import java.rmi.registry.Registry;
import java.rmi.registry.LocateRegistry;
import java.rmi.server.UnicastRemoteObject;

public class MessageServer implements MessageExchangeInterface
{
  private Hashtable<String, List<Message>> messages;

  public MessageServer()
  {
    messages = new Hashtable<>(); //create a hashtable for stoing messages
  }

  public void storeMessage(Message m)
  {
    List<Message> messageList = messages.get(m.getRecipient());
    if (messageList == null) //if this is the first message for this recipient,
                             //create a new message list.
    {
      messageList = new LinkedList<>();
      messages.put(
        m.getRecipient(),
        messageList
      );
    }
    messageList.add(m); //add the message to the messages
  }

  public Message accessNewestMessage(Message m)
  {
    Message r = null;//default return value
    List<Message> ms = messages.get(m.getRecipient());
    if (ms != null) //check that there are some messages
      r = ms.get(ms.size() - 1); //get the latest message
    return r;
  }

  public List<Message> accessMessages(Message m)
  {
    List<Message> ms = messages.get(m.getRecipient()); //get the list of messages for this recipient
    return ms;
  }

  static public void main(String[] args)
  {
    try
    {
      MessageServer server = new MessageServer();
      MessageExchangeInterface messageExchanger = 
        (MessageExchangeInterface) UnicastRemoteObject
                                    .exportObject(server, 0);

      Registry registry = LocateRegistry.getRegistry();
      registry.rebind("MessageService12345", messageExchanger); //rebind this server to the registry
                                                                //makes rerunning the server easier

      System.out.println("Message Server Ready");
    }
    catch (Exception e)
    {
      System.err.println("Server exception: " + e.toString());
      e.printStackTrace();
    }
  }
}
