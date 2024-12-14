# vision_language_project


## Environment

- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- timm
- transformers
- h5py

## Directory Setup

<!---------------------------------------------------------------------------------------------------------------->
The structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataset/`
    - `bertvocab/`
        - `v2` : bert tokernizer
    - `EndoVis-18-VQLA/` : seq_{1,2,3,4,5,6,7,9,10,11,12,14,15,16}. Each sequence folder follows the same folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from EndoVIS18 challange.
            - `vqla`
                - Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                    - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
        - `....`
        - `seq_16`
    - `EndoVis-17-VQLA/` : selected 97 frames from EndoVIS17 challange for external validation. 
        - `left_frames`
        - `vqla`
            - Q&A pairs and bounding box label.
            - `img_features`: Contains img_features extracted from each frame with different patch size.
                - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
    - `featre_extraction/`:
        - `feature_extraction_EndoVis18-VQLA-frcnn.py`: Used to extract features with Fast-RCNN and ResNet101.
        - `feature_extraction_EndoVis18-VQLA-resnet`: Used to extract features with ResNet18 (based on patch size).
- `models/`: 
    - GatedLanguageVisualEmbedding.py : GLVE module for visual and word embeddings and fusion.
    - LViTPrediction.py : our proposed LViT model for VQLA task.
    - VisualBertResMLP.py : VisualBERT ResMLP encoder from Surgical-VQA.
    - visualBertPrediction.py : VisualBert encoder-based model for VQLA task.
    - VisualBertResMLPPrediction.py : VisualBert ResMLP encoder-based model for VQLA task.
- dataloader.py
- train.py
- utils.py

---

## Dataset

[Link] (https://drive.google.com/drive/folders/10kFZFX0RwTosEwUAGNcKKUkb499hGHMj?usp=sharing)
<!-- 1. EndoVis-18-VQA (Image frames can be downloaded directly from EndoVis Challenge Website)
    - [VQLA](https://drive.google.com/file/d/1m7CSNY9PcUoCAUO_DoppDCi_l2L2RiFN/view?usp=sharing)
2. EndoVis-17-VLQA (External Validation Set)
    - [Images & VQLA](https://drive.google.com/file/d/1PQ-SDxwiNXs5nmV7PuBgBUlfaRRQaQAU/view?usp=sharing)   -->

---

### Run training

- Train on EndoVis-18-VLQA 
    ```bash
    python train.py --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver lvit --batch_size 8 --epochs 80
    ```

---

## Evaluation

- Evaluate on both EndoVis-18-VLQA & EndoVis-17-VLQA
    ```bash
    python train.py --validate True --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver lvit --batch_size 8
    ```

---

Chat

New Conversation

🤓 Explain a complex thing

Explain Artificial Intelligence so that I can explain it to my six-year-old child.


🧠 Get suggestions and create new ideas

Please give me the best 10 travel ideas around the world


💭 Translate, summarize, fix grammar and more…

Translate "I love you" French


GPT-3.5
Hello, how can I help you today?
blind tcp session hijacking
GPT-3.5
Blind TCP session hijacking is a type of attack where an attacker intercepts and takes control of a TCP session between two parties without their knowledge. This attack is considered "blind" because the attacker does not need to see or understand the actual data being transmitted.
During blind TCP session hijacking, the attacker guesses or predicts the sequence numbers used in TCP packets to impersonate one of the communicating parties. By injecting malicious packets with spoofed source IP addresses and correctly predicted sequence numbers, they can take over the session and potentially gain unauthorized access to sensitive information.
To protect against blind TCP session hijacking, various security measures can be implemented such as using strong encryption protocols (e.g., TLS), implementing firewall rules to block suspicious traffic, regularly updating software and systems to patch vulnerabilities, and employing intrusion detection/prevention systems.
Here are some links that provide more information about blind TCP session hijacking:

OWASP - Session Hijacking: https://owasp.org/www-community/attacks/Session_hijacking_attack

SANS Institute - Blind Hijackings: https://www.sans.org/reading-room/whitepapers/testing/blind-hijackings-33308

Imperva - Session Hijacking: https://www.imperva.com/learn/application-security/session-hijacking/

PortSwigger Web Security Academy - Session Management Attacks: https://portswigger.net/web-security/session-management


Please note that while these resources provide valuable insights into blind TCP session hijacking, it's essential to use this knowledge responsibly and for ethical purposes only.
what is TLS
GPT-3.5
TLS stands for Transport Layer Security. It is a cryptographic protocol designed to provide secure communication over the internet. TLS is the successor to SSL (Secure Sockets Layer) and is commonly used to encrypt data transmission between web browsers and servers, email clients and servers, virtual private networks (VPNs), and other applications that require secure communication.
TLS ensures the confidentiality, integrity, and authenticity of data by encrypting it during transmission. It prevents eavesdropping, tampering, and impersonation attacks.
Here are some key features of TLS:

Encryption: TLS uses symmetric encryption algorithms (such as AES) to encrypt the actual data being transmitted between two parties. It also employs an asymmetric encryption algorithm (such as RSA or Diffie-Hellman) for key exchange and negotiation.

Authentication: TLS provides mutual authentication between the client and server using digital certificates. This ensures that both parties can verify each other's identities before establishing a secure connection.

Data Integrity: TLS uses message authentication codes (MACs) to ensure that transmitted data has not been altered or tampered with during transit.

Forward Secrecy: TLS supports forward secrecy by generating unique session keys for each session established between a client and server. Even if one session's key is compromised, it does not affect the security of previous or future sessions.

Version Compatibility: TLS supports various versions such as TLS 1.0, 1.1, 1.2, 1.3 (the latest version). The choice of version depends on the capabilities supported by both client and server applications.


TLS plays a vital role in securing sensitive information transmitted over networks like passwords, credit card details, personal information in web applications or any other application communicating via TCP/IP protocols.
It's important for organizations to keep their systems up-to-date with the latest recommended versions of TLS to take advantage of its improved security features.
what is SYN flooding attack
GPT-3.5
SYN flooding attack, also known as a TCP SYN flood attack, is a type of denial-of-service (DoS) attack that targets the TCP three-way handshake process. The goal of this attack is to overwhelm the target server's resources and make it unable to handle legitimate client requests.
In a normal TCP connection establishment, the client sends a SYN (synchronize) packet to the server, and the server responds with a SYN-ACK (synchronize-acknowledge) packet. The client then sends an ACK (acknowledge) packet, and the connection is established.
During a SYN flooding attack, an attacker sends a large number of spoofed SYN packets to the target server without completing the three-way handshake by sending ACK packets. The attacker spoofs or falsifies the source IP addresses in these packets so that they cannot be traced back.
The targeted server receives these spoofed SYN packets and allocates system resources to create half-open connections in anticipation of completing the handshake. However, since no ACK packets are received from fake IP addresses, these half-open connections remain open and consume system resources like memory or available slots in connection tables.
As more and more half-open connections accumulate on the target server due to repeated spoofed SYN requests, it eventually reaches its limit - often referred to as "backlog queue" saturation. At this point, legitimate clients trying to establish connections with the overloaded server may be denied service or experience severe delays due to resource exhaustion.
To mitigate SYN flooding attacks, network administrators can implement various countermeasures such as:

Implementing firewalls or routers that can detect and drop suspicious traffic patterns associated with SYN floods.

Configuring network devices to have lower timeouts for incomplete connections.

Deploying intrusion detection/prevention systems (IDS/IPS) capable of detecting and mitigating such attacks.

Enabling SYN cookies on servers: This technique allows servers under attack to generate specially crafted SYN-ACK packets that do not require the allocation of system resources until an ACK is received.

Distributing traffic across multiple servers using load balancing techniques.


It's important for organizations to have proper network security measures in place to protect against SYN flooding attacks and ensure the availability of their services.
how to find my IP address in Mac
GPT-3.5
To find your IP address on a Mac, you can follow these steps:

Open the "System Preferences" by clicking on the Apple menu in the top-left corner of your screen and selecting "System Preferences."

In the System Preferences window, click on the "Network" icon.

In the Network settings, you will see a list of network interfaces on the left-hand side. Select the active network interface that you are currently connected to (e.g., Wi-Fi or Ethernet).

Once you select your active network interface, you will see information about that connection on the right-hand side.

If you are connected via Wi-Fi, look for "Wi-Fi Address." This is your IP address for wireless connections.

If you are connected via Ethernet (wired connection), look for "IP Address." This is your IP address for wired connections.




Your IP address will be displayed next to either "Wi-Fi Address" or "IP Address" depending on how you are connected to the network.
Alternatively, you can also check your IP address using Terminal by following these steps:

Open Terminal by going to Applications > Utilities > Terminal.

In Terminal, type any of the following commands and press Enter:

For IPv4: ifconfig | grep inet

For IPv6: ifconfig | grep inet6




The output will display information about all available network interfaces and their respective IP addresses.
Please note that if multiple network interfaces are active (e.g., both Wi-Fi and Ethernet), each interface may have its own unique IP address.
Your usage limit has been reached. Please upgrade your plan to continue using. Upgrade
GPT-4o Mini
coin image
10
Upgrade



Ask me anything...



Make a Review & Earn Credit ❤
Chat
Ask
Search
Write
Image
ChatPDF
Vision
Full Page
Invite & Earn
