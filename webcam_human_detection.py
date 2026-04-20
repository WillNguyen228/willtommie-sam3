# --- SAM3 Webcam Human Detection Script ---
import torch
import cv2
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

def draw_detections(frame, boxes, scores, labels=None, confidence_threshold=0.3):
    """
    Draw bounding boxes and masks on the frame.
    
    Args:
        frame: OpenCV frame (BGR format)
        boxes: Detection boxes
        scores: Confidence scores
        labels: Optional labels for each detection
        confidence_threshold: Minimum confidence to display
    
    Returns:
        frame with drawn detections
    """
    if boxes is None or len(boxes) == 0:
        return frame
    
    # Convert frame to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        if score.item() < confidence_threshold:
            continue
            
        # Extract box coordinates
        x1, y1, x2, y2 = [int(v.item()) for v in box]
        
        # Draw rectangle
        color = (0, 255, 0)  # Green for human detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence score
        label = f"Human: {score.item():.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background for text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def main():
    print("="*70)
    print("SAM3 Webcam Human Detection")
    print("="*70)
    print("\nInitializing...")
    
    # Load SAM3 model
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.3)
    print("✓ Model loaded successfully!")
    
    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    
    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✓ Webcam opened successfully!")
    print("\n" + "="*70)
    print("CONTROLS:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press '+' to increase detection speed")
    print("  - Press '-' to decrease detection speed")
    print("="*70 + "\n")
    
    # Initialize variables
    inference_state = None
    last_detection_time = 0
    detection_interval = 0.2  # Process detection every 0.2 seconds (faster!)
    frame_count = 0
    saved_count = 0
    process_width = 480  # Resize frame width for faster processing
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to grab frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Process detection at specified interval
            if current_time - last_detection_time >= detection_interval:
                # Resize frame for faster processing
                h, w = frame.shape[:2]
                scale = process_width / w
                resized_frame = cv2.resize(frame, (process_width, int(h * scale)))
                
                # Convert frame to PIL Image (RGB)
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Set image for SAM3 processing
                inference_state = processor.set_image(pil_image)
                
                # Reset prompts and segment with text prompt "person" or "human"
                processor.reset_all_prompts(inference_state)
                output = processor.set_text_prompt(
                    state=inference_state, 
                    prompt="person"  # You can also try "human", "man", "woman", "people"
                )
                
                # Get detection results
                boxes = output.get("boxes", None)
                scores = output.get("scores", [])
                
                # Scale boxes back to original frame size
                if boxes is not None and len(boxes) > 0:
                    boxes = boxes / scale
                
                last_detection_time = current_time
                
                # Print detection info
                if boxes is not None and len(scores) > 0:
                    num_detections = sum(1 for s in scores if s.item() > 0.3)
                    if num_detections > 0:
                        print(f"Frame {frame_count}: Detected {num_detections} person(s)")
            
            # Draw detections on frame
            if boxes is not None and len(boxes) > 0:
                frame = draw_detections(frame, boxes, scores, confidence_threshold=0.3)
            
            # Display FPS and detection interval
            actual_fps = 1.0 / detection_interval
            fps_text = f"Detection FPS: {actual_fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            interval_text = f"Interval: {detection_interval:.2f}s"
            cv2.putText(frame, interval_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('SAM3 Human Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"human_detection_{saved_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Saved: {filename}")
                saved_count += 1
            elif key == ord('+') or key == ord('='):
                # Increase detection speed (decrease interval)
                detection_interval = max(0.1, detection_interval - 0.05)
                print(f"Detection interval: {detection_interval:.2f}s ({1.0/detection_interval:.1f} FPS)")
            elif key == ord('-') or key == ord('_'):
                # Decrease detection speed (increase interval)
                detection_interval = min(2.0, detection_interval + 0.05)
                print(f"Detection interval: {detection_interval:.2f}s ({1.0/detection_interval:.1f} FPS)")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Done!")


if __name__ == "__main__":
    main()
