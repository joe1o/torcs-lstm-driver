import os
import msgParser
import carState
import carControl
import numpy as np
import torch
import pickle
from collections import deque

# Import custom model
from model import TorcsLSTM

class Driver(object):
    def __init__(self, stage):
        self.stage = stage

        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.gear = 1  
        self.reverse_mode = False  

        self.log_filename = "telemetry_log.csv"
        file_exists = os.path.exists(self.log_filename)

        self.log_file = open(self.log_filename, "a")  # Append mode

        if not file_exists:
            track_headers = ",".join([f"track[{i+1}]" for i in range(19)])
            opponent_headers = ",".join([f"opponent[{i+1}]" for i in range(36)])  # Added opponent sensors
            self.log_file.write(f"SpeedX,SpeedY,SpeedZ,Angle,TrackPos,Steer,Gear,Accel,RPM,Brake,ReverseMode,{track_headers},{opponent_headers}\n")

        # Load AI model and scalers
        self.load_ai_model()
        
        # Buffer for sequence of states for LSTM input
        self.seq_length = 10
        self.state_buffer = deque(maxlen=self.seq_length)
        
        # Initialize the buffer with empty states
        for _ in range(self.seq_length):
            empty_state = np.zeros(len(self.scalers['feature_columns']))
            self.state_buffer.append(empty_state)

    def load_ai_model(self):
        """Load the trained model and scalers"""
        try:
            # Load model
            checkpoint = torch.load('torcs_model_best.pth', map_location=torch.device('cpu'))
            
            # Load scalers
            with open('torcs_scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)
            
            # Create a new model instance
            input_size = len(self.scalers['feature_columns'])
            self.model = TorcsLSTM(input_size=input_size, output_size=5)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set model to evaluation mode
            
            print("AI model and scalers loaded successfully")
        except Exception as e:
            print(f"Error loading AI model: {e}")
            print("Falling back to default controls")
            self.model = None
            self.scalers = None

    def init(self):
        self.angles = [0 for _ in range(19)]

        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5

        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)
        speed_x = self.state.getSpeedX()
        speed_y = self.state.getSpeedY()
        speed_z = self.state.getSpeedZ()
        rpm = self.state.getRpm()
        angle = self.state.angle
        track_pos = self.state.trackPos
        track = self.state.getTrack()
        opponents = self.state.getOpponents()

        # Default AI predictions
        predicted_steer, predicted_accel, predicted_brake, predicted_gear, predicted_reverse = 0.0, 0.0, 0.0, 1.0, 0.0

        # Get AI predictions if model is available
        if self.model is not None:
            # Create current state array
            current_state = []
            
            # Add features in the same order they were trained on
            for feature in self.scalers['feature_columns']:
                if feature == 'SpeedX':
                    current_state.append(speed_x)
                elif feature == 'SpeedY':
                    current_state.append(speed_y)
                elif feature == 'SpeedZ':
                    current_state.append(speed_z)
                elif feature == 'Angle':
                    current_state.append(angle)
                elif feature == 'TrackPos':
                    current_state.append(track_pos)
                elif feature == 'RPM':
                    current_state.append(rpm)
                elif feature.startswith('track['):
                    idx = int(feature.strip('track[]')) - 1
                    if 0 <= idx < len(track):
                        current_state.append(track[idx])
                    else:
                        current_state.append(0.0)  # Fallback
                elif feature.startswith('opponent['):
                    idx = int(feature.strip('opponent[]')) - 1
                    if 0 <= idx < len(opponents):
                        current_state.append(opponents[idx])
                    else:
                        current_state.append(200.0)  # Default value for missing opponent sensors
            
            # Add the current state to the buffer
            self.state_buffer.append(np.array(current_state))
            
            # Only make predictions when buffer is full
            if len(self.state_buffer) == self.seq_length:
                # Convert state buffer to numpy array
                state_sequence = np.array(list(self.state_buffer))
                
                # Normalize the input
                normalized_sequence = self.scalers['features'].transform(state_sequence)
                
                # Convert to PyTorch tensor and add batch dimension
                input_tensor = torch.FloatTensor(normalized_sequence).unsqueeze(0)
                
                # Get prediction from model
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                # Convert model output back to original scale
                prediction = self.scalers['targets'].inverse_transform(output.numpy())
                
                # Store predictions
                predicted_steer, predicted_accel, predicted_brake, predicted_gear, predicted_reverse = prediction[0]

        # Apply AI predictions
        self.steer = predicted_steer  
        self.accel = np.clip(predicted_accel, 0.0, 1.0)  
        self.brake = np.clip(predicted_brake, 0.0, 1.0)  
        self.gear = int(round(np.clip(predicted_gear, -1, 4)))  # Gear
        self.reverse_mode = predicted_reverse > 0.5  

        # Apply controls
        self.control.setSteer(self.steer)
        self.control.setAccel(self.accel)
        self.control.setBrake(self.brake)
        self.control.setGear(self.gear)

        # Log telemetry data including opponent sensors
        self.log_data(speed_x, speed_y, speed_z)

        # Print predictions (for debugging)
        if self.stage == 2:  # Only in race mode
            print(f"AI Steer: {predicted_steer:.3f}, AI Accel: {predicted_accel:.3f}, AI Brake: {predicted_brake:.3f}, AI Gear: {predicted_gear:.1f}, AI Reverse: {predicted_reverse:.3f}, Applied: Steer={self.steer:.3f}, Accel={self.accel:.3f}, Brake={self.brake:.3f}, Gear={self.gear}, Reverse={self.reverse_mode}")

        return self.control.toMsg()

    def log_data(self, speed_x, speed_y, speed_z):
        """Logs telemetry data including opponent sensors"""
        angle = self.state.angle
        track_pos = self.state.trackPos
        steer = self.control.getSteer()
        gear = self.gear
        accel = self.control.getAccel()
        rpm = self.state.getRpm()
        brake = self.brake
        reverse_mode = int(self.reverse_mode)
        track_sensors = ",".join(map(str, self.state.getTrack()))
        opponent_sensors = ",".join(map(str, self.state.getOpponents()))  # Added opponent sensors

        self.log_file.write(f"{speed_x},{speed_y},{speed_z},{angle},{track_pos},{steer},{gear},{accel},{rpm},{brake},{reverse_mode},{track_sensors},{opponent_sensors}\n")

    def onShutDown(self):
        self.log_file.close()

    def onRestart(self):
        """Reset state buffer on restart"""
        # Clear state buffer to reset LSTM context
        for _ in range(self.seq_length):
            empty_state = np.zeros(len(self.scalers['feature_columns']))
            self.state_buffer.append(empty_state)
        # Log the reset event
        self.log_data(self.state.getSpeedX(), self.state.getSpeedY(), self.state.getSpeedZ())