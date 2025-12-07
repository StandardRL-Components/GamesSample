import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:29:23.441194
# Source Brief: brief_00500.md
# Brief Index: 500
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where a robotic arm collects, combines, and lifts cubes.

    **Objective:** Combine cubes to reach a total weight of 500 and lift it
    into the target zone within 1200 steps.

    **Action Space (MultiDiscrete([5, 2, 2])):**
    - `action[0]` (Movement):
        - 0: No-op
        - 1: Up (Controls Joint 1, or Joint 3 if Shift is held)
        - 2: Down (Controls Joint 1, or Joint 3 if Shift is held)
        - 3: Left (Controls Joint 2)
        - 4: Right (Controls Joint 2)
    - `action[1]` (Space): 0=Released, 1=Held. Pressing toggles grab/release.
    - `action[2]` (Shift): 0=Released, 1=Held. Modifies the 'Up'/'Down' actions.

    **Reward Structure:**
    - +0.1 per grab.
    - +1.0 per successful combination.
    - +100 for winning (lifting >= 500 weight).
    - -100 for timing out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a robotic arm to pick up and combine cubes. "
        "Combine cubes to reach the target weight and lift it into the designated zone to win."
    )
    user_guide = (
        "Controls: Use arrow keys to move the arm. Hold Shift while pressing ↑/↓ to control the wrist. "
        "Press Space to grab, combine, or release cubes."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1200
    TARGET_WEIGHT = 500
    ARM_BASE_X, ARM_BASE_Y = WIDTH // 2, HEIGHT - 20
    ARM_SEGMENT_LENGTHS = [80, 70, 60]
    ARM_SPEED = 0.05  # Radians per step
    GRAB_RADIUS = 20
    TARGET_ZONE_RECT = pygame.Rect(WIDTH // 4, 10, WIDTH // 2, 80)

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_ARM = (180, 190, 200)
    COLOR_JOINT = (220, 230, 240)
    COLOR_TARGET_ZONE = (0, 100, 200, 50) # RGBA for transparency
    COLOR_TARGET_ZONE_ACTIVE = (100, 200, 255, 80)
    COLOR_TEXT = (240, 240, 240)
    WEIGHT_COLORS = {
        10: (50, 150, 255), 20: (50, 255, 150), 40: (255, 255, 100),
        80: (255, 150, 50), 160: (255, 80, 80), 320: (255, 100, 255),
        640: (255, 255, 255)
    }

    # --- Rewards ---
    REWARD_GRAB = 0.1
    REWARD_COMBINE = 1.0
    REWARD_WIN = 100.0
    REWARD_TIMEOUT = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_big = pygame.font.SysFont('Consolas', 48, bold=True)
        
        # State variables are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.joint_angles = None
        self.JOINT_LIMITS = None
        self.arm_positions = None
        self.cubes = None
        self.held_cube = None
        self.prev_space_held = None
        self.particles = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.joint_angles = np.array([-math.pi / 2, math.pi / 4, math.pi / 4], dtype=np.float32)
        self.JOINT_LIMITS = [
            (-math.pi, 0),          # Joint 1 (base)
            (0, math.pi * 0.8),     # Joint 2
            (-math.pi/2, math.pi/2) # Joint 3 (end effector)
        ]
        
        self.prev_space_held = False
        self.particles = []
        self._spawn_cubes()
        self.held_cube = None
        
        self._update_arm_kinematics()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        step_reward = 0.0

        # 1. Process Actions
        step_reward += self._handle_actions(action)

        # 2. Update Physics & Game State
        self._update_arm_kinematics()
        self._update_particles()
        self.steps += 1
        self.score = self.held_cube['weight'] if self.held_cube else 0

        # 3. Check for Termination
        terminated, terminal_reward = self._check_termination()
        step_reward += terminal_reward
        if terminated:
            self.game_over = True

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            step_reward += self.REWARD_TIMEOUT

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Joint Movement ---
        # Shift modifies which joint is controlled by Up/Down
        if movement in [1, 2]: # Up or Down
            joint_idx = 2 if shift_held else 0
            direction = -1 if movement == 1 else 1 # Up decreases angle from -pi/2
            self.joint_angles[joint_idx] += direction * self.ARM_SPEED
            self.joint_angles[joint_idx] = np.clip(
                self.joint_angles[joint_idx], self.JOINT_LIMITS[joint_idx][0], self.JOINT_LIMITS[joint_idx][1]
            )
        
        # Left/Right always control Joint 2
        if movement in [3, 4]: # Left or Right
            direction = 1 if movement == 3 else -1
            self.joint_angles[1] += direction * self.ARM_SPEED
            self.joint_angles[1] = np.clip(
                self.joint_angles[1], self.JOINT_LIMITS[1][0], self.JOINT_LIMITS[1][1]
            )

        # --- Grab/Release Action (on key press) ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            reward += self._refactored_grab_logic()
        self.prev_space_held = space_held
        
        return reward

    def _refactored_grab_logic(self):
        """A clearer implementation of the grab/combine/release logic."""
        end_effector_pos = self.arm_positions[3]
        
        # --- Case 1: Holding a cube ---
        if self.held_cube:
            # Check for combination: is there another cube nearby?
            target_for_combine, dist = self._find_closest_cube(end_effector_pos)
            if target_for_combine and dist < self.GRAB_RADIUS + target_for_combine['size'] / 2:
                # COMBINE
                new_weight = self.held_cube['weight'] + target_for_combine['weight']
                self.cubes.remove(target_for_combine)
                self.held_cube = self._create_cube(pos=end_effector_pos, weight=new_weight)
                self._create_particles(end_effector_pos, self.held_cube['color'], 50)
                return self.REWARD_COMBINE
            
            # No cube to combine with, so RELEASE
            else:
                # Check for win condition if in target zone
                if self.TARGET_ZONE_RECT.collidepoint(end_effector_pos):
                    if self.held_cube['weight'] >= self.TARGET_WEIGHT:
                        self.game_over = True # This will be caught by _check_termination
                        self._create_particles(end_effector_pos, self.held_cube['color'], 100)
                        self.score = self.held_cube['weight'] # Set score for info
                        self.held_cube = None
                        return 0 # Win reward handled later
                
                # Drop the cube (either failed win or normal drop)
                if self.held_cube:
                    self.held_cube['pos'] = end_effector_pos
                    self.cubes.append(self.held_cube)
                    self.held_cube = None
                return 0

        # --- Case 2: Not holding a cube ---
        else:
            # Attempt to GRAB
            target_to_grab, dist = self._find_closest_cube(end_effector_pos)
            if target_to_grab and dist < self.GRAB_RADIUS + target_to_grab['size'] / 2:
                self.held_cube = target_to_grab
                self.cubes.remove(target_to_grab)
                return self.REWARD_GRAB
        
        return 0 # No action taken

    def _check_termination(self):
        # Win condition is checked during the release action inside _refactored_grab_logic
        if self.game_over and self.score >= self.TARGET_WEIGHT:
            return True, self.REWARD_WIN
        # The timeout (truncation) is handled in step()
        return False, 0.0

    def _update_arm_kinematics(self):
        p0 = np.array([self.ARM_BASE_X, self.ARM_BASE_Y])
        a1, a2, a3 = self.joint_angles[0], self.joint_angles[1], self.joint_angles[2]
        
        p1 = p0 + self.ARM_SEGMENT_LENGTHS[0] * np.array([math.cos(a1), math.sin(a1)])
        
        cumulative_angle_2 = a1 + a2 - math.pi/2
        p2 = p1 + self.ARM_SEGMENT_LENGTHS[1] * np.array([math.cos(cumulative_angle_2), math.sin(cumulative_angle_2)])
        
        cumulative_angle_3 = cumulative_angle_2 + a3
        p3 = p2 + self.ARM_SEGMENT_LENGTHS[2] * np.array([math.cos(cumulative_angle_3), math.sin(cumulative_angle_3)])
        
        self.arm_positions = [p0, p1, p2, p3]

    def _spawn_cubes(self):
        self.cubes = []
        initial_weights = [10, 10, 10, 10, 20, 20, 40]
        for weight in initial_weights:
            pos = [
                self.np_random.uniform(50, self.WIDTH - 50),
                self.np_random.uniform(150, self.HEIGHT - 100)
            ]
            self.cubes.append(self._create_cube(pos, weight))

    def _create_cube(self, pos, weight):
        size = 5 + math.sqrt(weight) * 2
        # Find the closest base weight for color
        color_key = min(self.WEIGHT_COLORS.keys(), key=lambda k: abs(k - weight))
        color = self.WEIGHT_COLORS[color_key]
        return {'pos': list(pos), 'weight': weight, 'size': size, 'color': color}

    def _find_closest_cube(self, pos):
        closest_cube = None
        min_dist = float('inf')
        if not self.cubes:
            return None, min_dist
            
        for cube in self.cubes:
            dist = np.linalg.norm(np.array(pos) - np.array(cube['pos']))
            if dist < min_dist:
                min_dist = dist
                closest_cube = cube
        return closest_cube, min_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw target zone
        is_in_zone = self.TARGET_ZONE_RECT.collidepoint(self.arm_positions[3])
        zone_color = self.COLOR_TARGET_ZONE_ACTIVE if is_in_zone and self.held_cube else self.COLOR_TARGET_ZONE
        s = pygame.Surface(self.TARGET_ZONE_RECT.size, pygame.SRCALPHA)
        s.fill(zone_color)
        self.screen.blit(s, self.TARGET_ZONE_RECT.topleft)
        pygame.draw.rect(self.screen, (zone_color[0], zone_color[1], zone_color[2]), self.TARGET_ZONE_RECT, 2)


        # Draw floor cubes
        for cube in self.cubes:
            self._draw_cube(cube)

        # Draw arm
        for i in range(len(self.arm_positions) - 1):
            p_start = self.arm_positions[i]
            p_end = self.arm_positions[i+1]
            pygame.draw.aaline(self.screen, self.COLOR_ARM, p_start, p_end, 5)
        
        for pos in self.arm_positions:
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 8, self.COLOR_JOINT)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 8, self.COLOR_JOINT)
        
        # Draw end effector grab radius and held cube
        end_effector_pos = self.arm_positions[3]
        if self.held_cube:
            self.held_cube['pos'] = list(end_effector_pos)
            self._draw_cube(self.held_cube)
        else: # Show grab radius when empty
             pygame.gfxdraw.aacircle(self.screen, int(end_effector_pos[0]), int(end_effector_pos[1]), int(self.GRAB_RADIUS), (*self.COLOR_JOINT, 100))

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), p['color'])

    def _draw_cube(self, cube):
        pos, size, color = cube['pos'], cube['size'], cube['color']
        rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, width=2, border_radius=3)

    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {self.MAX_STEPS - self.steps}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (15, 15))

        # Target weight
        target_text = f"TARGET: {self.TARGET_WEIGHT}"
        target_surf = self.font_ui.render(target_text, True, self.COLOR_TEXT)
        self.screen.blit(target_surf, (self.WIDTH - target_surf.get_width() - 15, 15))

        # Current held weight
        current_weight = self.held_cube['weight'] if self.held_cube else 0
        weight_color = self.held_cube['color'] if self.held_cube else self.COLOR_TEXT
        weight_text = f"HELD: {current_weight}"
        weight_surf = self.font_ui.render(weight_text, True, weight_color)
        self.screen.blit(weight_surf, (self.WIDTH - weight_surf.get_width() - 15, 40))

        if self.game_over:
            if self.score >= self.TARGET_WEIGHT:
                end_text = "SUCCESS!"
                end_color = (100, 255, 100)
            else:
                end_text = "TIME UP!"
                end_color = (255, 100, 100)
            
            end_surf = self.font_big.render(end_text, True, end_color)
            self.screen.blit(end_surf, end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "held_weight": self.held_cube['weight'] if self.held_cube else 0,
            "cubes_on_floor": len(self.cubes)
        }
        
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifetime': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['lifetime'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # --- Manual Play Example ---
    # Un-comment the os.environ line below to run with a display
    # os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # This part is for human play, so we need a real display.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] != "dummy":
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Robo-Lifter")
        clock = pygame.time.Clock()
        
        done = False
        total_reward = 0
        
        print("\n--- Manual Control ---")
        print("Arrows: Move Arm Joints 1 & 2")
        print("Shift + Up/Down: Move Arm Joint 3")
        print("Space: Grab / Combine / Release")
        print("----------------------\n")

        while not done:
            # Convert observation to a Pygame surface
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Action Mapping for Manual Control ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if reward != 0:
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Held: {info['held_weight']}")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            clock.tick(30) # Limit to 30 FPS for playability

        print(f"\nGame Over! Final Score (Total Reward): {total_reward:.2f}")
        env.close()
    else:
        print("Skipping manual play because SDL_VIDEODRIVER is 'dummy'.")
        print("Set SDL_VIDEODRIVER to a valid backend (e.g., 'x11') to play manually.")
        # A simple test to ensure the environment runs headless
        print("Running a short headless test...")
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()
        print("Headless test complete.")