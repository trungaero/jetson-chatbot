# Copy scripts into project folder
cp start-gemma.sh ~/chatbot-jetson/jetson-chatbot/start-gemma.sh
cp start.sh ~/chatbot-jetson/jetson-chatbot/start.sh
chmod +x ~/chatbot-jetson/jetson-chatbot/start-gemma.sh ~/chatbot-jetson/jetson-chatbot/start.sh

# Install both systemd services
sudo cp gemma-server.service /etc/systemd/system/gemma-server@trung.service
sudo cp jetson-chatbot.service /etc/systemd/system/jetson-chatbot@trung.service

sudo systemctl daemon-reload

# Enable both — gemma-server will always start before jetson-chatbot
sudo systemctl enable gemma-server@trung
sudo systemctl enable jetson-chatbot@trung