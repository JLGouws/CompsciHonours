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

  private void sendMessage()
  {
    Scanner sc = new Scanner(System.in);
    System.out.println("Please enter a recipient.");
    String r = sc.next();
    System.out.println("Please enter a message.");
    String b = "";
    while (sc.hasNextLine())
      b += sc.nextLine() + "\n";

    this.sendMessage(r, b);
    System.out.println("Message sent");
  }

  private void retrieveLatest()
  {
    Message m = this.accessLatest();
    if (m == null)
      System.out.println("You have no messages.");
    else
      System.out.println(m);
    Scanner sc = new Scanner(System.in);
    sc.nextLine();
  }

  private static void printMenu()
  {
    System.out.println(
      "Would you like to:\n" +
      "Send a message? [s]\n" +
      "View most recent message? [m]\n" +
      "Quit? [q]"
    );
  }

  static public void main(String[] args)
  {
    Scanner sc = new Scanner(System.in);
    System.out.println("Please enter your username.");
    String u = sc.next(),
           cmd;
    MessageClient mc = new MessageClient(u);

    boolean running = true;

    while (running)
    {
      printMenu();
      cmd = sc.next();
      switch (cmd)
      {
        case "s":
          mc.sendMessage();
          break;
        case "m":
          mc.retrieveLatest();
          break;
        case "q":
          running = false;
          break;
        default:
          System.out.println("You entered an invalid command");
      }
    }
    
  }
}
