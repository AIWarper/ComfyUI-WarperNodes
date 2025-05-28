# ComfyUI/custom_nodes/ComfyUI_WarperNodes/warper_nodes.py

import torch
import numpy as np
import json
import copy
import cv2

# --- Keypoint Indices (COCO Format) ---
NOSE = 0; NECK = 1; R_SHOULDER = 2; R_ELBOW = 3; R_WRIST = 4; L_SHOULDER = 5; L_ELBOW = 6; L_WRIST = 7
R_HIP = 8; R_KNEE = 9; R_ANKLE = 10; L_HIP = 11; L_KNEE = 12; L_ANKLE = 13; R_EYE = 14; L_EYE = 15
R_EAR = 16; L_EAR = 17

# --- Drawing Constants (Adapted from typical OpenPose/DWPose visualizations) ---
BODY_LIMB_THICKNESS_SCALE = 256
BODY_POINT_RADIUS_SCALE = 128
FACE_HAND_POINT_RADIUS_SCALE_FACTOR = 1.5

RED = (0, 0, 255); GREEN = (0, 255, 0); BLUE = (255, 0, 0); YELLOW = (0, 255, 255); CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255); WHITE = (255, 255, 255); BLACK = (0, 0, 0); ORANGE = (0, 165, 255); PURPLE = (128, 0, 128)

BODY_DRAW_CONFIG = [
    ((NOSE, NECK), PURPLE),
    ((NECK, R_SHOULDER), GREEN), ((R_SHOULDER, R_ELBOW), GREEN), ((R_ELBOW, R_WRIST), GREEN),
    ((NECK, L_SHOULDER), BLUE), ((L_SHOULDER, L_ELBOW), BLUE), ((L_ELBOW, L_WRIST), BLUE),
    ((NECK, R_HIP), RED), ((R_HIP, R_KNEE), RED), ((R_KNEE, R_ANKLE), RED),
    ((NECK, L_HIP), ORANGE), ((L_HIP, L_KNEE), ORANGE), ((L_KNEE, L_ANKLE), ORANGE),
    # ((R_HIP, L_HIP), WHITE), # Removed pelvis line
    ((NOSE, R_EYE), YELLOW), ((R_EYE, R_EAR), YELLOW),
    ((NOSE, L_EYE), CYAN), ((L_EYE, L_EAR), CYAN),
]

FACE_CONNECTIONS = [
    list(range(0, 17)), list(range(17, 22)), list(range(22, 27)), list(range(27, 31)), list(range(31, 36)),
    list(range(36, 42)) + [36], list(range(42, 48)) + [42], list(range(48, 60)) + [48], list(range(60, 68)) + [60]
]
FACE_COLOR = (180, 180, 180)

HAND_CONNECTIONS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
]
HAND_L_COLOR = (0, 220, 220); HAND_R_COLOR = (220, 0, 220)


def draw_keypoints_and_connections(canvas, keypoints_array, connections, color_map_or_default_color, thickness, point_radius, is_body=False):
    if keypoints_array.size == 0:
        return

    if isinstance(connections, list) and connections:
        if isinstance(connections[0], list): # Path-based connections (e.g., FACE_CONNECTIONS, HAND_CONNECTIONS)
            for path in connections:
                for i in range(len(path) - 1):
                    p1_idx, p2_idx = path[i], path[i+1]
                    pt1 = get_point(keypoints_array, p1_idx)
                    pt2 = get_point(keypoints_array, p2_idx)
                    if pt1 is not None and pt2 is not None:
                        cv2.line(canvas, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color_map_or_default_color, thickness=thickness)
        elif isinstance(connections[0], tuple) and len(connections[0]) == 2 and isinstance(connections[0][0], tuple): # Limb-based connections with specific colors (e.g., BODY_DRAW_CONFIG)
             for (p1_idx, p2_idx), color_val in connections: # connections is BODY_DRAW_CONFIG
                pt1 = get_point(keypoints_array, p1_idx)
                pt2 = get_point(keypoints_array, p2_idx)
                if pt1 is not None and pt2 is not None:
                    cv2.line(canvas, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color_val, thickness=thickness)

    # Draw keypoints
    for i in range(keypoints_array.shape[0]):
        pt = get_point(keypoints_array, i)
        if pt is not None:
            point_color_to_use = WHITE # Default for non-body points or if color not found for body
            if is_body and isinstance(color_map_or_default_color, list): # color_map_or_default_color is BODY_DRAW_CONFIG for body
                # Try to color body points based on one of their limb connections
                # This is a simplification; could be more sophisticated
                found_color = False
                for (p1_idx_conn, p2_idx_conn), color_val_conn in color_map_or_default_color:
                    if p1_idx_conn == i or p2_idx_conn == i:
                        point_color_to_use = color_val_conn
                        found_color = True
                        break
                # if not found_color and i == NOSE: point_color_to_use = PURPLE # Nose is often colored specifically
            elif not is_body: # For face/hands, use the default color passed
                point_color_to_use = color_map_or_default_color

            cv2.circle(canvas, (pt[0], pt[1]), radius=point_radius, color=point_color_to_use, thickness=-1)

def draw_scaled_pose(people_data, canvas_height, canvas_width):
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    if not people_data: return canvas

    # Dynamic scaling of drawing parameters
    line_thickness = max(1, int(canvas_height / BODY_LIMB_THICKNESS_SCALE))
    body_point_rad = max(1, int(canvas_height / BODY_POINT_RADIUS_SCALE))
    face_hand_point_rad = max(1, int(body_point_rad / FACE_HAND_POINT_RADIUS_SCALE_FACTOR)) # Smaller radius for face/hand

    for person in people_data:
        body_kp_flat = person.get("pose_keypoints_2d", [])
        if body_kp_flat:
            body_kp = np.array(body_kp_flat).reshape(-1, 3)
            draw_keypoints_and_connections(canvas, body_kp, BODY_DRAW_CONFIG, BODY_DRAW_CONFIG, line_thickness, body_point_rad, is_body=True)

        face_kp_flat = person.get("face_keypoints_2d", [])
        if face_kp_flat:
            face_kp = np.array(face_kp_flat).reshape(-1, 3)
            draw_keypoints_and_connections(canvas, face_kp, FACE_CONNECTIONS, FACE_COLOR, max(1, line_thickness // 2), face_hand_point_rad)

        hand_l_kp_flat = person.get("hand_left_keypoints_2d", [])
        if hand_l_kp_flat:
            hand_l_kp = np.array(hand_l_kp_flat).reshape(-1, 3)
            draw_keypoints_and_connections(canvas, hand_l_kp, HAND_CONNECTIONS, HAND_L_COLOR, max(1, line_thickness // 2), face_hand_point_rad)

        hand_r_kp_flat = person.get("hand_right_keypoints_2d", [])
        if hand_r_kp_flat:
            hand_r_kp = np.array(hand_r_kp_flat).reshape(-1, 3)
            draw_keypoints_and_connections(canvas, hand_r_kp, HAND_CONNECTIONS, HAND_R_COLOR, max(1, line_thickness // 2), face_hand_point_rad)
    return canvas

def get_point_raw_coords(keypoints_array, index):
    # Check if keypoints_array is 2D, index is valid, and confidence is above threshold
    if keypoints_array.ndim == 2 and 0 <= index < keypoints_array.shape[0] and keypoints_array[index, 2] > 0.01: # Confidence threshold
        return keypoints_array[index, :2].astype(float) # Return x, y as float
    return None

def get_point(keypoints_array, index):
    pt_coords = get_point_raw_coords(keypoints_array, index)
    return pt_coords.astype(int) if pt_coords is not None else None


def set_point(keypoints_array, index, new_coords_np_array, confidence_override=None):
    if keypoints_array.ndim == 2 and 0 <= index < keypoints_array.shape[0]:
        keypoints_array[index, :2] = new_coords_np_array # Set x, y
        if confidence_override is not None:
            keypoints_array[index, 2] = confidence_override
        # If original confidence was very low (effectively not detected) and we're setting non-zero coords,
        # set a default high confidence.
        elif keypoints_array[index, 2] < 0.01 and (new_coords_np_array[0] != 0 or new_coords_np_array[1] != 0):
            keypoints_array[index, 2] = 1.0


class DWPoseScalerNode_Warper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "neck_length_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "torso_adjust_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}), # In pixels
                "head_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "arm_length_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "hand_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "leg_length_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "keep_feet_grounded": ("BOOLEAN", {"default": True}),
                "reference_frame_index": ("INT", {"default": 0, "min": 0}), # Frame index for reference measurements
            }
        }
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("scaled_image", "scaled_pose_keypoints")
    FUNCTION = "scale_pose"
    CATEGORY = "Warper Tools"

    def _get_original_measurements(self, ref_person_data, canvas_width, canvas_height):
        measurements = {}
        body_kp_flat = ref_person_data.get("pose_keypoints_2d", [])
        body_kp_ref = np.array(body_kp_flat).reshape(-1, 3).astype(float) if body_kp_flat else np.empty((0,3), dtype=float)

        if body_kp_ref.size == 0: return None

        l_ankle_pt_coords = get_point_raw_coords(body_kp_ref, L_ANKLE)
        r_ankle_pt_coords = get_point_raw_coords(body_kp_ref, R_ANKLE)
        measurements["original_l_ankle_y"] = l_ankle_pt_coords[1] if l_ankle_pt_coords is not None else None
        measurements["original_r_ankle_y"] = r_ankle_pt_coords[1] if r_ankle_pt_coords is not None else None
        return measurements

    def scale_pose(self, pose_keypoints, neck_length_scale, torso_adjust_y, head_scale, arm_length_scale, hand_scale, leg_length_scale, keep_feet_grounded, reference_frame_index):
        if not pose_keypoints or not isinstance(pose_keypoints, list) or not pose_keypoints[0] or not isinstance(pose_keypoints[0], dict) or "canvas_height" not in pose_keypoints[0]:
            print(f"Warper DWPoseScaler: Invalid or empty pose_keypoints input. Returning blank image.")
            dummy_img = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
            return (dummy_img, [])

        modified_frames_data = []
        output_images_np_list = []

        # Determine reference frame for grounding if keep_feet_grounded is True
        reference_frame_index = min(max(0, reference_frame_index), len(pose_keypoints) - 1)
        ref_frame_data = pose_keypoints[reference_frame_index]

        ref_person_measurements = None
        if ref_frame_data.get("people") and len(ref_frame_data["people"]) > 0:
            # Assuming first person in reference frame for measurements
            ref_person_measurements = self._get_original_measurements(ref_frame_data["people"][0], ref_frame_data.get("canvas_width", 512),ref_frame_data.get("canvas_height", 512))

        original_ref_l_ankle_y, original_ref_r_ankle_y = None, None
        if ref_person_measurements:
            original_ref_l_ankle_y = ref_person_measurements.get("original_l_ankle_y")
            original_ref_r_ankle_y = ref_person_measurements.get("original_r_ankle_y")


        for frame_idx, frame_data_orig_outer in enumerate(pose_keypoints):
            frame_data = copy.deepcopy(frame_data_orig_outer) # Work on a deep copy to avoid modifying original input
            canvas_height = frame_data.get("canvas_height", 512)
            canvas_width = frame_data.get("canvas_width", 512)

            if "people" not in frame_data or not frame_data["people"]:
                img_np = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) # Blank image for frames with no people
                output_images_np_list.append(img_np)
                modified_frames_data.append(frame_data)
                continue

            for person_idx, person in enumerate(frame_data["people"]):
                # Ensure keypoints are numpy arrays of floats for calculations
                body_kp = np.array(person.get("pose_keypoints_2d", [])).reshape(-1, 3).astype(float) if person.get("pose_keypoints_2d") else np.empty((0,3), dtype=float)
                face_kp = np.array(person.get("face_keypoints_2d", [])).reshape(-1, 3).astype(float) if person.get("face_keypoints_2d") else np.empty((0,3), dtype=float)
                hand_l_kp = np.array(person.get("hand_left_keypoints_2d", [])).reshape(-1,3).astype(float) if person.get("hand_left_keypoints_2d") else np.empty((0,3), dtype=float)
                hand_r_kp = np.array(person.get("hand_right_keypoints_2d", [])).reshape(-1,3).astype(float) if person.get("hand_right_keypoints_2d") else np.empty((0,3), dtype=float)

                if body_kp.size == 0: continue # Skip person if no body keypoints

                # Store initial ankle Y positions for this frame for grounding, if available
                initial_l_ankle_y_for_grounding = get_point_raw_coords(body_kp, L_ANKLE)[1] if get_point_raw_coords(body_kp, L_ANKLE) is not None else None
                initial_r_ankle_y_for_grounding = get_point_raw_coords(body_kp, R_ANKLE)[1] if get_point_raw_coords(body_kp, R_ANKLE) is not None else None

                # --- 1. Leg Length Scaling ---
                if leg_length_scale != 1.0:
                    for prefix, hip_idx, knee_idx, ankle_idx in [("L", L_HIP, L_KNEE, L_ANKLE), ("R", R_HIP, R_KNEE, R_ANKLE)]:
                        current_hip_coords = get_point_raw_coords(body_kp, hip_idx)
                        current_knee_coords_orig = get_point_raw_coords(body_kp, knee_idx) # Knee before thigh scaling
                        current_ankle_coords = get_point_raw_coords(body_kp, ankle_idx)

                        if keep_feet_grounded: # Scale upwards from ankle
                            if current_ankle_coords is None or current_knee_coords_orig is None: continue
                            current_shin_vec = current_ankle_coords - current_knee_coords_orig # Vector from original knee to ankle
                            scaled_shin_vec = current_shin_vec * leg_length_scale
                            new_knee_pt = current_ankle_coords - scaled_shin_vec # New knee is scaled shin away from fixed ankle
                            set_point(body_kp, knee_idx, new_knee_pt)

                            current_knee_coords_new = get_point_raw_coords(body_kp, knee_idx) # Knee after shin scaling
                            if current_knee_coords_new is None or current_hip_coords is None: continue
                            current_thigh_vec_from_orig_knee = current_knee_coords_orig - current_hip_coords # Vector from hip to original knee
                            scaled_thigh_vec = current_thigh_vec_from_orig_knee * leg_length_scale
                            new_hip_pt = current_knee_coords_new - scaled_thigh_vec # New hip is scaled thigh away from new knee
                            set_point(body_kp, hip_idx, new_hip_pt)
                        else: # Scale downwards from hip (feet not grounded)
                            if current_hip_coords is None or current_knee_coords_orig is None: continue
                            current_thigh_vec = current_knee_coords_orig - current_hip_coords
                            scaled_thigh_vec = current_thigh_vec * leg_length_scale
                            new_knee_pt = current_hip_coords + scaled_thigh_vec
                            set_point(body_kp, knee_idx, new_knee_pt)

                            current_knee_coords_new = get_point_raw_coords(body_kp, knee_idx) # Knee after thigh scaling
                            if current_knee_coords_new is None or current_ankle_coords is None: continue # Ankle needs original knee
                            current_shin_vec_from_orig_ankle = current_ankle_coords - current_knee_coords_orig # Vector from original knee to original ankle
                            scaled_shin_vec = current_shin_vec_from_orig_ankle * leg_length_scale
                            new_ankle_pt = current_knee_coords_new + scaled_shin_vec
                            set_point(body_kp, ankle_idx, new_ankle_pt)

                # --- 2. Foot Grounding (Post-Leg Scaling Y-Offset) ---
                # This step applies an overall Y-shift to the whole pose
                # to bring the feet back to their original (or reference frame's) Y-positions.
                if keep_feet_grounded and leg_length_scale != 1.0: # Only apply if legs were scaled
                    y_offset_total = 0.0
                    num_ankles_for_offset = 0

                    # Determine target Y for grounding: current frame's initial Y if available, else reference frame's Y
                    target_l_ankle_y = initial_l_ankle_y_for_grounding if initial_l_ankle_y_for_grounding is not None else original_ref_l_ankle_y
                    target_r_ankle_y = initial_r_ankle_y_for_grounding if initial_r_ankle_y_for_grounding is not None else original_ref_r_ankle_y

                    new_l_ankle_coords = get_point_raw_coords(body_kp, L_ANKLE)
                    if target_l_ankle_y is not None and new_l_ankle_coords is not None:
                        y_offset_total += (target_l_ankle_y - new_l_ankle_coords[1])
                        num_ankles_for_offset +=1
                    new_r_ankle_coords = get_point_raw_coords(body_kp, R_ANKLE)
                    if target_r_ankle_y is not None and new_r_ankle_coords is not None:
                        y_offset_total += (target_r_ankle_y - new_r_ankle_coords[1])
                        num_ankles_for_offset +=1

                    if num_ankles_for_offset > 0:
                        final_y_offset = y_offset_total / num_ankles_for_offset
                        # Apply this offset to ALL keypoints for this person
                        for kp_array_curr in [body_kp, face_kp, hand_l_kp, hand_r_kp]:
                            if kp_array_curr.size == 0: continue
                            for i in range(kp_array_curr.shape[0]):
                                if kp_array_curr[i,2] > 0.01: # Only shift visible points
                                    kp_array_curr[i, 1] += final_y_offset
                
                # --- 3. Neck Length Scaling (Option C revised: Upper Neck Stretch without intrinsic head scaling) ---
                if neck_length_scale != 1.0:
                    current_neck_coords_anchor_body = get_point_raw_coords(body_kp, NECK)
                    if current_neck_coords_anchor_body is not None:
                        original_nose_coords = get_point_raw_coords(body_kp, NOSE)
                        if original_nose_coords is not None:
                            orig_vec_neck_to_nose = original_nose_coords - current_neck_coords_anchor_body
                            scaled_vec_neck_to_nose = orig_vec_neck_to_nose * neck_length_scale
                            new_nose_coords = current_neck_coords_anchor_body + scaled_vec_neck_to_nose
                            
                            head_body_dependents_indices_rel_to_nose = [R_EYE, L_EYE, R_EAR, L_EAR]
                            orig_vectors_from_orig_nose_to_body_dependents = {}
                            for kp_idx in head_body_dependents_indices_rel_to_nose:
                                dependent_pt_coords = get_point_raw_coords(body_kp, kp_idx)
                                if dependent_pt_coords is not None:
                                    orig_vectors_from_orig_nose_to_body_dependents[kp_idx] = dependent_pt_coords - original_nose_coords
                                else:
                                    orig_vectors_from_orig_nose_to_body_dependents[kp_idx] = None
                            
                            orig_vectors_from_orig_nose_to_face_kps = []
                            if face_kp.size > 0:
                                for i in range(face_kp.shape[0]):
                                    dependent_face_pt_coords = get_point_raw_coords(face_kp, i)
                                    if dependent_face_pt_coords is not None:
                                        orig_vectors_from_orig_nose_to_face_kps.append(dependent_face_pt_coords - original_nose_coords)
                                    else:
                                        orig_vectors_from_orig_nose_to_face_kps.append(None)

                            set_point(body_kp, NOSE, new_nose_coords)
                            current_new_nose_coords_after_set = get_point_raw_coords(body_kp, NOSE) 

                            if current_new_nose_coords_after_set is not None:
                                for kp_idx, orig_vec_from_nose in orig_vectors_from_orig_nose_to_body_dependents.items():
                                    if orig_vec_from_nose is not None:
                                        new_dependent_pos = current_new_nose_coords_after_set + orig_vec_from_nose
                                        set_point(body_kp, kp_idx, new_dependent_pos)
                                
                                if face_kp.size > 0:
                                    for i, orig_vec_from_nose in enumerate(orig_vectors_from_orig_nose_to_face_kps):
                                        if orig_vec_from_nose is not None:
                                            new_dependent_pos = current_new_nose_coords_after_set + orig_vec_from_nose
                                            set_point(face_kp, i, new_dependent_pos)

                # --- 4. Arm Length Scaling ---
                if arm_length_scale != 1.0:
                    for prefix, shoulder_idx, elbow_idx, wrist_idx in [("L", L_SHOULDER, L_ELBOW, L_WRIST), ("R", R_SHOULDER, R_ELBOW, R_WRIST)]:
                        current_shoulder_coords = get_point_raw_coords(body_kp, shoulder_idx)
                        current_elbow_coords_orig = get_point_raw_coords(body_kp, elbow_idx) # Elbow before upper arm scaling
                        current_wrist_coords_orig = get_point_raw_coords(body_kp, wrist_idx) # Wrist before forearm scaling

                        if current_shoulder_coords is None: continue

                        new_elbow_coords_for_forearm = current_elbow_coords_orig # Default if elbow not scaled

                        # Scale Upper Arm (Shoulder to Elbow)
                        if current_elbow_coords_orig is not None:
                            current_upper_arm_vec = current_elbow_coords_orig - current_shoulder_coords
                            scaled_upper_arm_vec = current_upper_arm_vec * arm_length_scale
                            new_elbow_pt = current_shoulder_coords + scaled_upper_arm_vec
                            set_point(body_kp, elbow_idx, new_elbow_pt)
                            new_elbow_coords_for_forearm = get_point_raw_coords(body_kp, elbow_idx) # Use new elbow for forearm

                        # Scale Forearm (Elbow to Wrist)
                        if new_elbow_coords_for_forearm is not None and current_wrist_coords_orig is not None:
                            # We need the original elbow to calculate the original forearm vector
                            if current_elbow_coords_orig is not None:
                                current_forearm_vec = current_wrist_coords_orig - current_elbow_coords_orig
                                scaled_forearm_vec = current_forearm_vec * arm_length_scale
                                new_wrist_pt = new_elbow_coords_for_forearm + scaled_forearm_vec # Attach to newly scaled elbow
                                set_point(body_kp, wrist_idx, new_wrist_pt)


                # --- 5. Hand Scaling ---
                # Scales hand keypoints relative to their respective (body) wrist keypoint.
                # The body wrist keypoint's position is determined by arm scaling.
                # The hand's own base (keypoint 0) is effectively re-anchored to the body wrist.
                if hand_scale != 1.0:
                    for wrist_idx, current_hand_kp_array in [(L_WRIST, hand_l_kp), (R_WRIST, hand_r_kp)]:
                        current_body_wrist_coords = get_point_raw_coords(body_kp, wrist_idx) # This is the (potentially scaled) body wrist

                        if current_body_wrist_coords is not None and current_hand_kp_array.size > 0:
                            current_hand_base_coords = get_point_raw_coords(current_hand_kp_array, 0) # Hand's own base point (usually wrist)

                            if current_hand_base_coords is not None:
                                for i in range(current_hand_kp_array.shape[0]): # Iterate all hand keypoints
                                    current_finger_kp_coords = get_point_raw_coords(current_hand_kp_array, i)
                                    if current_finger_kp_coords is not None:
                                        # Vector from the hand's own base to the current finger keypoint
                                        vec_from_current_hand_base_to_finger = current_finger_kp_coords - current_hand_base_coords
                                        scaled_vec_finger_kp = vec_from_current_hand_base_to_finger * hand_scale
                                        # New position is the scaled vector added to the BODY wrist (effectively moving and scaling the hand)
                                        new_finger_kp_pos = current_body_wrist_coords + scaled_vec_finger_kp
                                        set_point(current_hand_kp_array, i, new_finger_kp_pos)

                # --- 6. Head Size Scaling ---
                # This scales head components (Eyes, Ears on body_kp, and all face_kp)
                # around the NOSE keypoint. The NOSE and NECK positions are those
                # potentially modified by the neck_length_scale step.
                if head_scale != 1.0:
                    # Neck and Nose positions are now those potentially modified by neck_length_scale (head stretch)
                    current_neck_for_headscale = get_point_raw_coords(body_kp, NECK) # This is the original NECK position (anchor for head stretch)
                    current_nose_for_headscale = get_point_raw_coords(body_kp, NOSE) # This is the NOSE after head stretch

                    if current_neck_for_headscale is not None and current_nose_for_headscale is not None:
                        # Scale Eyes and Ears (on body_kp) relative to the current NOSE
                        head_body_indices_to_scale_around_nose = [R_EYE, L_EYE, R_EAR, L_EAR] # NOSE is the center of scaling here
                        for kp_idx in head_body_indices_to_scale_around_nose:
                            current_component_coords = get_point_raw_coords(body_kp, kp_idx)
                            if current_component_coords is not None:
                                vec_curr_nose_to_curr_component = current_component_coords - current_nose_for_headscale
                                scaled_vec_component_from_nose = vec_curr_nose_to_curr_component * head_scale
                                new_component_pt_val = current_nose_for_headscale + scaled_vec_component_from_nose
                                set_point(body_kp, kp_idx, new_component_pt_val)

                        # Scale face_kp relative to the current NOSE
                        if face_kp.size > 0:
                            for i in range(face_kp.shape[0]):
                                current_face_kp_coords = get_point_raw_coords(face_kp, i)
                                if current_face_kp_coords is not None:
                                    vec_curr_nose_to_curr_face_kp = current_face_kp_coords - current_nose_for_headscale
                                    scaled_vec_face_kp_from_nose = vec_curr_nose_to_curr_face_kp * head_scale
                                    new_face_kp_val = current_nose_for_headscale + scaled_vec_face_kp_from_nose
                                    set_point(face_kp, i, new_face_kp_val)

                # --- 7. Torso Y-Adjustment ---
                # This shifts the upper body (neck, shoulders, head, arms, hands) vertically.
                # Hips and legs are NOT affected by this specific adjustment.
                if torso_adjust_y != 0.0:
                    y_offset = float(torso_adjust_y)

                    # Store the state of keypoints *before* this torso shift, but *after* all other scalings
                    body_kp_orig_state = body_kp.copy()
                    face_kp_orig_state = face_kp.copy()
                    hand_l_kp_orig_state = hand_l_kp.copy()
                    hand_r_kp_orig_state = hand_r_kp.copy()

                    # Get original positions of anchors before they are moved
                    orig_neck_for_torso_adjust = get_point_raw_coords(body_kp_orig_state, NECK)
                    orig_l_shoulder_for_torso_adjust = get_point_raw_coords(body_kp_orig_state, L_SHOULDER)
                    orig_r_shoulder_for_torso_adjust = get_point_raw_coords(body_kp_orig_state, R_SHOULDER)

                    # 1. Move the primary torso anchors (Neck, Shoulders)
                    for kp_idx in [NECK, L_SHOULDER, R_SHOULDER]:
                        orig_anchor_pos = get_point_raw_coords(body_kp_orig_state, kp_idx) # Position from after scaling, before this shift
                        if orig_anchor_pos is not None:
                            new_y = orig_anchor_pos[1] + y_offset
                            set_point(body_kp, kp_idx, np.array([orig_anchor_pos[0], new_y]))
                    
                    # Get the new positions of these anchors *after* the shift
                    new_neck_pos_torso_adj = get_point_raw_coords(body_kp, NECK)
                    new_l_shoulder_pos_torso_adj = get_point_raw_coords(body_kp, L_SHOULDER)
                    new_r_shoulder_pos_torso_adj = get_point_raw_coords(body_kp, R_SHOULDER)

                    # 2. Move dependents relative to their respective new anchor positions
                    # Head parts (body_kp subset and all face_kp) move with the NECK
                    if orig_neck_for_torso_adjust is not None and new_neck_pos_torso_adj is not None:
                        # Head parts on body_kp
                        for kp_idx in [NOSE, R_EYE, L_EYE, R_EAR, L_EAR]:
                            orig_part_pos = get_point_raw_coords(body_kp_orig_state, kp_idx)
                            if orig_part_pos is not None:
                                vec_orig_neck_to_part = orig_part_pos - orig_neck_for_torso_adjust
                                set_point(body_kp, kp_idx, new_neck_pos_torso_adj + vec_orig_neck_to_part)
                        # All face_kp points
                        if face_kp_orig_state.size > 0:
                            for i in range(face_kp_orig_state.shape[0]):
                                orig_part_pos = get_point_raw_coords(face_kp_orig_state, i)
                                if orig_part_pos is not None:
                                    vec_orig_neck_to_part = orig_part_pos - orig_neck_for_torso_adjust
                                    set_point(face_kp, i, new_neck_pos_torso_adj + vec_orig_neck_to_part)
                    
                    # Left arm parts (Elbow, Wrist on body_kp) move with L_SHOULDER
                    if orig_l_shoulder_for_torso_adjust is not None and new_l_shoulder_pos_torso_adj is not None:
                        for kp_idx in [L_ELBOW, L_WRIST]:
                            orig_part_pos = get_point_raw_coords(body_kp_orig_state, kp_idx)
                            if orig_part_pos is not None:
                                vec_orig_shoulder_to_part = orig_part_pos - orig_l_shoulder_for_torso_adjust
                                set_point(body_kp, kp_idx, new_l_shoulder_pos_torso_adj + vec_orig_shoulder_to_part)
                    
                    # Right arm parts (Elbow, Wrist on body_kp) move with R_SHOULDER
                    if orig_r_shoulder_for_torso_adjust is not None and new_r_shoulder_pos_torso_adj is not None:
                        for kp_idx in [R_ELBOW, R_WRIST]:
                            orig_part_pos = get_point_raw_coords(body_kp_orig_state, kp_idx)
                            if orig_part_pos is not None:
                                vec_orig_shoulder_to_part = orig_part_pos - orig_r_shoulder_for_torso_adjust
                                set_point(body_kp, kp_idx, new_r_shoulder_pos_torso_adj + vec_orig_shoulder_to_part)

                    # Hands (hand_l_kp, hand_r_kp) move with their respective WRISTS
                    # Need to use wrist positions *after* they've been moved by the shoulder y-adjust
                    orig_l_wrist_for_torso_adjust = get_point_raw_coords(body_kp_orig_state, L_WRIST) # Wrist pos BEFORE this y-adjust
                    new_l_wrist_pos_torso_adj = get_point_raw_coords(body_kp, L_WRIST)             # Wrist pos AFTER L_SHOULDER moved it

                    if orig_l_wrist_for_torso_adjust is not None and new_l_wrist_pos_torso_adj is not None and hand_l_kp_orig_state.size > 0:
                        for i in range(hand_l_kp_orig_state.shape[0]):
                            orig_hand_kp_pos = get_point_raw_coords(hand_l_kp_orig_state, i)
                            if orig_hand_kp_pos is not None:
                                vec_orig_body_wrist_to_hand_kp = orig_hand_kp_pos - orig_l_wrist_for_torso_adjust
                                set_point(hand_l_kp, i, new_l_wrist_pos_torso_adj + vec_orig_body_wrist_to_hand_kp)

                    orig_r_wrist_for_torso_adjust = get_point_raw_coords(body_kp_orig_state, R_WRIST) # Wrist pos BEFORE this y-adjust
                    new_r_wrist_pos_torso_adj = get_point_raw_coords(body_kp, R_WRIST)             # Wrist pos AFTER R_SHOULDER moved it
                    if orig_r_wrist_for_torso_adjust is not None and new_r_wrist_pos_torso_adj is not None and hand_r_kp_orig_state.size > 0:
                        for i in range(hand_r_kp_orig_state.shape[0]):
                            orig_hand_kp_pos = get_point_raw_coords(hand_r_kp_orig_state, i)
                            if orig_hand_kp_pos is not None:
                                vec_orig_body_wrist_to_hand_kp = orig_hand_kp_pos - orig_r_wrist_for_torso_adjust
                                set_point(hand_r_kp, i, new_r_wrist_pos_torso_adj + vec_orig_body_wrist_to_hand_kp)


                # --- Finalize and Clip ---
                # Clip all keypoint coordinates to be within canvas boundaries.
                # Using float for min/max arrays for np.clip compatibility with keypoint arrays.
                a_min_kps = np.array([0, 0, 0], dtype=float) # min x, y, confidence
                a_max_kps = np.array([canvas_width - 1, canvas_height - 1, 1.0], dtype=float) # max x, y, confidence

                if body_kp.size > 0:
                    person["pose_keypoints_2d"] = np.clip(body_kp, a_min_kps, a_max_kps).astype(float).flatten().tolist()
                if face_kp.size > 0:
                    person["face_keypoints_2d"] = np.clip(face_kp, a_min_kps, a_max_kps).astype(float).flatten().tolist()
                if hand_l_kp.size > 0:
                    person["hand_left_keypoints_2d"] = np.clip(hand_l_kp, a_min_kps, a_max_kps).astype(float).flatten().tolist()
                if hand_r_kp.size > 0:
                    person["hand_right_keypoints_2d"] = np.clip(hand_r_kp, a_min_kps, a_max_kps).astype(float).flatten().tolist()

            # Draw the modified pose onto a new canvas for this frame
            img_np = draw_scaled_pose(frame_data.get("people", []), canvas_height, canvas_width)
            output_images_np_list.append(img_np)
            modified_frames_data.append(frame_data) # Add the modified frame data (with people keypoints)

        # Convert list of numpy image arrays to a torch tensor
        output_images_torch = torch.from_numpy(np.array(output_images_np_list).astype(np.float32) / 255.0)

        # Handle edge case of empty output or single image needing batch dimension
        if output_images_torch.ndim == 3 and len(output_images_np_list) == 1 : # Single image, add batch dim
             output_images_torch = output_images_torch.unsqueeze(0)
        elif output_images_torch.ndim == 0 and not output_images_np_list: # No images processed, create empty tensor
             # Use canvas dimensions from the first frame of input if available, else default
             final_canvas_h = pose_keypoints[0].get("canvas_height", 256) if pose_keypoints else 256
             final_canvas_w = pose_keypoints[0].get("canvas_width", 256) if pose_keypoints else 256
             output_images_torch = torch.empty((0, final_canvas_h, final_canvas_w, 3), dtype=torch.float32)

        return (output_images_torch, modified_frames_data)