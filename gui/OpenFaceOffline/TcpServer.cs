using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenFaceOffline
{
    using System;
    using System.Net;
    using System.Net.Sockets;
    using System.Text;
    using System.Threading;

    using System;
    using System.Net;
    using System.Net.Sockets;
    using System.Text;

    public class TcpServer
    {
        private TcpListener listener;
        private TcpClient client;
        volatile bool done = false;


        public TcpServer(int portNum)
        {
            listener = new TcpListener(IPAddress.Any, portNum);
        }

        public void Start()
        {
            done = false;
            client = null;
            listener.Start();
            Thread listenerThread = new Thread(new ThreadStart(DoLoop));
            listenerThread.Start();
        }

        public void Stop()
        {
            done = true;
            listener.Stop();
            if(client!=null)
            {
                client.Close();
            }
            client = null;
        }

        private void DoLoop()
        {
            try
            {
                while (!done)
                {
                    Console.Write("Waiting for connection...");
                    client = listener.AcceptTcpClient();

                    Console.WriteLine("Connection accepted.");
                    break;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("tcpServer listener interrupted");
            }
        }

        public void Send(String msg)
        {
            if(client==null)
            {
                Console.WriteLine("No TCP Client connectet...");
                return;
            }
            try
            {
                NetworkStream ns=client.GetStream();
                byte[] msgBytes = System.Text.Encoding.ASCII.GetBytes(msg);
                ns.Write(msgBytes, 0, msgBytes.Length);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());

                Stop();
                Start();
            }
        }

    }

}
