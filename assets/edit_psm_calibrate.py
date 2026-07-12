import numpy as np
from scipy.spatial.transform import Rotation

def edit_psm_calibration(input_file, output_file):
    # 1. Load the original .npz file
    data = np.load(input_file)
    
    # 2. Extract all arrays into a mutable dictionary
    # data.files contains the names of the arrays ('PSM1', 'PSM2', 'BASE1', 'BASE2')
    arrays_dict = {key: data[key] for key in data.files}
    
    # Optional: Print the original state of PSM1
    print("--- Original PSM2 ---")
    print(arrays_dict['PSM2'])
    
    # ==========================================
    # 3. MAKE YOUR EDITS HERE
    # ==========================================
    
    # --- PSM2: correct POSITION-DEPENDENT (rotation) errors ---
    #
    # The observed errors grow with position, so they are a rotation of the
    # calibration, NOT a constant offset. Editing only [i,3] (translation) shifts
    # every prediction by the same amount and cannot remove a gradient.
    #
    # Observations, in the prediction/world frame
    #   (x = +x lateral, y = y, z = away from the camera):
    #     err1: +0.03 m in z  ->  prediction drifts -0.007 m in y   (z<->y)
    #     err2: +0.10 m in x  ->  prediction drifts +0.010 m in z   (x<->z, further)
    #
    # Each gradient becomes a small rotation (rad). If, after re-testing, one
    # axis's error gets BIGGER instead of smaller, flip that term's sign (your
    # world frame's z may point toward the camera, or x/y may be swapped).
    # wx =  0.003 / 0.03    # z<->y coupling   (~ +0.233 rad)
    wx =  0.000 / 0.03    # z<->y coupling   (~ +0.233 rad)
    # wy = -0.010 / 0.10    # x<->z coupling   (~ -0.100 rad)
    wy = -0.000 / 0.10    # x<->z coupling   (~ -0.100 rad)
    wz =  0.0
    C = Rotation.from_rotvec(-np.array([wx, wy, wz])).as_matrix()  # inverse = correction

    T2 = arrays_dict['PSM2']
    T2[:3, :3] = C @ T2[:3, :3]
    T2[:3,  3] = C @ T2[:3,  3]

    # Constant-offset touch-up: after the rotation nulled the gradient, the
    # gripper is a uniform +1 cm toward +y, so shift predictions -1 cm in y.
    # arrays_dict['PSM2'][1, 3] -= 0.010
    # arrays_dict['PSM2'][0, 3] += 0.003   # x offset, if needed
    # arrays_dict['PSM2'][2, 3] += 0.006   # z offset, if needed
    
    # Optional: Print the modified state of PSM1
    print("\n--- Modified PSM2 ---")
    print(arrays_dict['PSM2'])
    
    # print("--- Original PSM1 ---")
    # print(arrays_dict['PSM1'])

    # arrays_dict['PSM1'][0, 3] += 0.001
    # arrays_dict['PSM1'][2, 3] += 0.002

    # print("\n--- Modified PSM1 ---")
    # print(arrays_dict['PSM1'])
    # ==========================================
    
    # 4. Save the updated dictionary back to an .npz file
    # Using **unpacking to pass the dictionary keys as keyword arguments
    np.savez(output_file, **arrays_dict)
    
    # 5. Close the original NpzFile to free up memory
    data.close()
    
    print(f"\nSuccessfully saved the modified data to '{output_file}'")

if __name__ == "__main__":
    # Define your input and output filenames
    # You can set output_filename = 'psm_calibration.npz' to overwrite the original,
    # but it is usually safer to write to a new file first.
    input_filename = 'psm_calibration.npz'
    output_filename = 'psm_calibration_edited.npz'
    
    edit_psm_calibration(input_filename, output_filename)
