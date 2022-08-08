import java.util.Scanner;

import java.applet.*;

import java.rmi.RemoteException;
import java.rmi.registry.Registry;
import java.rmi.registry.LocateRegistry;

public class MessageClient
{
  MessageExchangeInterface messageExchanger;
  private final String name;

  private MessageClient(String name)
  {
    try
    {
      messageExchanger =
        (MessageExchangeInterface)
          LocateRegistry
            .getRegistry()
              .lookup("MessageService12345");
    }
    catch (Exception e)
    {
      System.err.println("Client could not find server:" + e.toString());
      e.printStackTrace();
    }

    this.name = name;
  }

  private void sendMessage(String to, String content)
  {
    Message m = new Message(name, to, content);
    try
    {
      messageExchanger.storeMessage(m);
    }
    catch (RemoteException rmEx)
    {
      System.err.println("MessageClient for " + name + "could not store message: " + rmEx);
      rmEx.printStackTrace();
    }
  }

  private Message accessLatest()
  {
    Message m = null;
    try
    {
      m = messageExchanger.
              accessNewestMessage
              (
                new Message(name)
              );
    }
    catch (RemoteException rmEx)
    {
      System.err.println("MessageClient for " + name + " could not access message: " + rmEx);
      rmEx.printStackTrace();
    }
    return m; 
  }

  static public void main(String[] args)
  {
    Scanner sc = new Scanner(System.in);
    System.out.println("Please enter your username.");
    String u = sc.next();
    MessageClient mc = new MessageClient(u);
    System.out.println("Please enter a recipient.");
    String r = sc.next();
    System.out.println("Please enter a message.");
    String b = sc.next();
    mc.sendMessage(r, b);
    
    System.out.println(mc.accessLatest());
  }
}
