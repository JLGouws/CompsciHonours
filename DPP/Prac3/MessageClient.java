import java.util.List;
import java.util.Scanner;
import java.util.ListIterator;

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
              .lookup("MessageService12345"); //find server in registry
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
    try //the server might run into an issue
    {
      messageExchanger.storeMessage(m); //store message on server
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
    try //the server might run into an issue
    {
      m = messageExchanger.
              accessNewestMessage //get message from server
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

  private List<Message> accessAll()
  {
    List<Message> m = null;
    try //the server might run into an issue
    {
      m = messageExchanger.
            accessMessages
              (
                new Message(name) //set message recipient
              );
    }
    catch (RemoteException rmEx)
    {
      System.err.println("MessageClient for " + name + " could not access messages: " + rmEx);
      rmEx.printStackTrace();
    }
    return m; 
  }

  private void sendMessage()
  {
    Scanner sc = new Scanner(System.in);
    System.out.println("\033[H\033[2JPlease enter a recipient."); //get message recipient
    String r = sc.next();
    System.out.println("Please enter a message.");
    String b = "";
    while (sc.hasNextLine())//^D to send message on unix systems
      b += sc.nextLine() + "\n";//get the next line of message

    this.sendMessage(r, b);
    System.out.println("\033[H\033[2JMessage sent. Press enter to return to the main menu.");
    sc.nextLine();
  }

  private void retrieveLatest()
  {
    Message m = this.accessLatest(); //get the latest message from server
    if (m == null)
      System.out.println("You have no messages."); //no messages
    else
      {
        System.out.println("Here is the message. Press enter to return to the main menu.");
        System.out.println(m);
      }
    Scanner sc = new Scanner(System.in);
    sc.nextLine();
  }

  private static void printNextMessageActions()
  {
    System.out.println(//some ui goodness
      "\033[H\033[2JWould you like to:\n" +
      "View previous mesage? [p]\n" +
      "Return to menu? [r]"
    );
  }

  private void getAll()
  {
    List<Message> ms = this.accessAll(); //get messages from server
    Scanner sc = new Scanner(System.in);
    if (ms == null || ms.size() == 0)
      System.out.println("You have no messages.");
    else
      { ListIterator<Message> msi = ms.listIterator(ms.size());
        String cmd;
        System.out.print("\033[H\033[2J");
        while (msi.hasPrevious()) //work through list of messages
          { System.out.println("Here is the message. Press enter for follwing actions.");
            System.out.println(msi.previous()); //print the message
            sc.nextLine();
            printNextMessageActions();
            cmd = sc.nextLine();
            switch (cmd)
            {
              case "p":
                System.out.print("\033[H\033[2J");
                break;
              case "r":
                return;
              default:
                msi.next(); //print the same message
                System.out.println("\033[H\033[2JYou entered an invalid command:\n");
            }
          }
        System.out.println("\033[H\033[2JThere are no more messages. Press enter to return to the main menu.");
      }
      sc.nextLine();
  }

  private static void printMenu()
  {
    System.out.println(
      "\033[H\033[2JWould you like to:\n" +
      "Send a message? [s]\n" +
      "View most recent message? [m]\n" +
      "View all recent messages? [a]\n" +
      "Quit? [q]"
    );
  }

  static public void main(String[] args)
  {
    Scanner sc = new Scanner(System.in);
    System.out.println("\033[H\033[2JPlease enter your username.");//get user's name
    String u = sc.next(),
           cmd;
    MessageClient mc = new MessageClient(u);

    boolean running = true;

    while (running)
    {
      printMenu(); //print out readable menu
      cmd = sc.next();
      switch (cmd)//check user's input
      {
        case "s":
          mc.sendMessage();
          break;
        case "m":
          mc.retrieveLatest();
          break;
        case "a":
          mc.getAll();
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
