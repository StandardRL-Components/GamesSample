import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:10:04.497415
# Source Brief: brief_00308.md
# Brief Index: 308
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls two robotic arms to assemble widgets.
    The agent must learn to position both arms correctly and trigger an assembly action
    to succeed. The challenge lies in coordinating the two arms based on a visual blueprint.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) for the selected arm.
    - actions[1]: Space button (0=released, 1=held) to trigger the assembly attempt.
    - actions[2]: Shift button (0=released, 1=held) to toggle which arm is selected.
    
    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - Potential-based shaping reward for moving arms closer to their targets.
    - +10 for a successful assembly.
    - -5 for a failed assembly attempt.
    - +50 for completing all 5 widgets.
    - -0.01 per step to encourage efficiency.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Control two robotic arms to assemble widgets according to a blueprint. "
        "Position each arm correctly and trigger the assembly action to score points."
    )
    user_guide = (
        "Use the arrow keys to move the selected arm. Press shift to switch between arms. "
        "Press space to attempt assembly."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game Constants ---
        self.MAX_STEPS = 2000
        self.WIDGETS_TO_WIN = 5
        self.ARM_SEGMENT_LENGTH = 70
        self.ARM_BASE_1 = (self.WIDTH // 2 - 120, self.HEIGHT // 2 + 80)
        self.ARM_BASE_2 = (self.WIDTH // 2 + 120, self.HEIGHT // 2 + 80)
        self.ANGLE_INCREMENT = 2.5  # degrees
        self.ASSEMBLY_TOLERANCE = 5.0 # degrees
        self.JOINT1_MIN, self.JOINT1_MAX = -160, -20
        self.JOINT2_MIN, self.JOINT2_MAX = -90, 90
        self.VISUAL_LERP_RATE = 0.25 # For smooth animation

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 55)
        self.COLOR_PLATFORM = (40, 55, 75)
        self.COLOR_ARM1 = (60, 180, 255)
        self.COLOR_ARM2 = (255, 160, 50)
        self.COLOR_ARM1_ACTIVE_GLOW = (160, 220, 255)
        self.COLOR_ARM2_ACTIVE_GLOW = (255, 200, 120)
        self.COLOR_BLUEPRINT = (60, 70, 85)
        self.COLOR_SUCCESS = (100, 255, 150)
        self.COLOR_FAIL = (255, 100, 100)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # --- State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.widgets_assembled = None
        self.active_arm = None
        self.last_shift_held = None
        self.last_space_held = None
        self.particles = None
        self.last_assembly_status = None
        self.assembly_feedback_timer = None
        self.arm1_joints = None
        self.arm2_joints = None
        self.visual_arm1_joints = None
        self.visual_arm2_joints = None
        self.target_arm1_joints = None
        self.target_arm2_joints = None
        
        # self.reset() # No need to call reset in init
        # self.validate_implementation() # No need for validation call in prod
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.widgets_assembled = 0
        self.active_arm = 0  # 0 for arm 1, 1 for arm 2
        self.last_shift_held = False
        self.last_space_held = False
        self.particles = []
        self.last_assembly_status = None
        self.assembly_feedback_timer = 0

        # Initialize arm joint angles (state)
        self.arm1_joints = np.array([-90.0, 0.0])
        self.arm2_joints = np.array([-90.0, 0.0])
        
        # Visual angles for smooth interpolation
        self.visual_arm1_joints = self.arm1_joints.copy()
        self.visual_arm2_joints = self.arm2_joints.copy()
        
        self._generate_new_target()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small time penalty
        self.steps += 1

        # --- Action Handling ---
        # 1. Handle arm selection toggle (on press)
        if shift_held and not self.last_shift_held:
            self.active_arm = 1 - self.active_arm
        self.last_shift_held = shift_held

        # 2. Handle arm movement
        prev_dist = self._calculate_total_distance(self.arm1_joints, self.arm2_joints)
        
        active_joints = self.arm1_joints if self.active_arm == 0 else self.arm2_joints
        if movement == 1: active_joints[0] -= self.ANGLE_INCREMENT # Up
        elif movement == 2: active_joints[0] += self.ANGLE_INCREMENT # Down
        elif movement == 3: active_joints[1] -= self.ANGLE_INCREMENT # Left
        elif movement == 4: active_joints[1] += self.ANGLE_INCREMENT # Right
        
        # Clamp angles to their limits
        active_joints[0] = np.clip(active_joints[0], self.JOINT1_MIN, self.JOINT1_MAX)
        active_joints[1] = np.clip(active_joints[1], self.JOINT2_MIN, self.JOINT2_MAX)

        # 3. Calculate potential-based reward for movement
        new_dist = self._calculate_total_distance(self.arm1_joints, self.arm2_joints)
        reward += prev_dist - new_dist

        # 4. Handle assembly trigger (on press)
        if space_held and not self.last_space_held:
            # SFX: Assembly_Attempt.wav
            is_success = self._check_assembly_success()
            if is_success:
                reward += 10
                self.score += 10
                self.widgets_assembled += 1
                self.last_assembly_status = 'success'
                self.assembly_feedback_timer = 30
                # SFX: Success_Chime.wav
                midpoint = self._get_assembly_point()
                self._create_particles(midpoint, self.COLOR_SUCCESS, 40, 4.0)
                if self.widgets_assembled < self.WIDGETS_TO_WIN:
                    self._generate_new_target()
            else:
                reward -= 5
                self.score -= 5
                self.last_assembly_status = 'fail'
                self.assembly_feedback_timer = 30
                # SFX: Fail_Buzzer.wav
                midpoint = self._get_assembly_point()
                self._create_particles(midpoint, self.COLOR_FAIL, 20, 2.0)
        self.last_space_held = space_held

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.widgets_assembled >= self.WIDGETS_TO_WIN:
            reward += 50
            self.score += 50
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _get_observation(self):
        # Interpolate visual angles towards state angles for smooth animation
        self.visual_arm1_joints += (self.arm1_joints - self.visual_arm1_joints) * self.VISUAL_LERP_RATE
        self.visual_arm2_joints += (self.arm2_joints - self.visual_arm2_joints) * self.VISUAL_LERP_RATE
        
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "widgets_assembled": self.widgets_assembled,
            "active_arm": self.active_arm,
            "arm1_dist_to_target": np.sum(np.abs(self.arm1_joints - self.target_arm1_joints)),
            "arm2_dist_to_target": np.sum(np.abs(self.arm2_joints - self.target_arm2_joints)),
        }

    # --- Helper Methods ---

    def _generate_new_target(self):
        """Generates a new, reachable target configuration for both arms."""
        self.target_arm1_joints = np.array([
            self.np_random.uniform(self.JOINT1_MIN + 10, self.JOINT1_MAX - 10),
            self.np_random.uniform(self.JOINT2_MIN + 10, self.JOINT2_MAX - 10)
        ])
        self.target_arm2_joints = np.array([
            self.np_random.uniform(self.JOINT1_MIN + 10, self.JOINT1_MAX - 10),
            self.np_random.uniform(self.JOINT2_MIN + 10, self.JOINT2_MAX - 10)
        ])

    def _calculate_total_distance(self, arm1j, arm2j):
        """Calculates the sum of angular distances from targets for all joints."""
        dist1 = np.sum(np.abs(arm1j - self.target_arm1_joints))
        dist2 = np.sum(np.abs(arm2j - self.target_arm2_joints))
        return dist1 + dist2

    def _check_assembly_success(self):
        """Checks if both arms are within tolerance of their targets."""
        err1 = np.abs(self.arm1_joints - self.target_arm1_joints)
        err2 = np.abs(self.arm2_joints - self.target_arm2_joints)
        return np.all(err1 < self.ASSEMBLY_TOLERANCE) and np.all(err2 < self.ASSEMBLY_TOLERANCE)

    def _get_arm_endpoints(self, base_pos, joints):
        """Calculates the x,y coordinates of arm segments."""
        angle1_rad = math.radians(joints[0])
        angle2_rad = math.radians(joints[0] + joints[1])
        
        p1 = (
            base_pos[0] + self.ARM_SEGMENT_LENGTH * math.cos(angle1_rad),
            base_pos[1] + self.ARM_SEGMENT_LENGTH * math.sin(angle1_rad)
        )
        p2 = (
            p1[0] + self.ARM_SEGMENT_LENGTH * math.cos(angle2_rad),
            p1[1] + self.ARM_SEGMENT_LENGTH * math.sin(angle2_rad)
        )
        return p1, p2
    
    def _get_assembly_point(self):
        """Calculates the midpoint between the two arm tips."""
        _, p1_tip = self._get_arm_endpoints(self.ARM_BASE_1, self.visual_arm1_joints)
        _, p2_tip = self._get_arm_endpoints(self.ARM_BASE_2, self.visual_arm2_joints)
        return ((p1_tip[0] + p2_tip[0]) / 2, (p1_tip[1] + p2_tip[1]) / 2)

    # --- Rendering Methods ---

    def _render_game(self):
        self._draw_background_and_platform()
        self._draw_blueprint()
        
        # Draw Arms
        self._draw_arm(self.ARM_BASE_1, self.visual_arm1_joints, self.COLOR_ARM1, 
                       self.COLOR_ARM1_ACTIVE_GLOW, is_active=(self.active_arm == 0))
        self._draw_arm(self.ARM_BASE_2, self.visual_arm2_joints, self.COLOR_ARM2, 
                       self.COLOR_ARM2_ACTIVE_GLOW, is_active=(self.active_arm == 1))
        
        self._update_and_draw_particles()

    def _draw_background_and_platform(self):
        # Draw grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))
        
        # Draw assembly platform
        platform_rect = pygame.Rect(self.WIDTH // 2 - 200, self.HEIGHT // 2 + 70, 400, 50)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform_rect, border_top_left_radius=8, border_top_right_radius=8)
        pygame.draw.line(self.screen, self.COLOR_GRID, platform_rect.topleft, platform_rect.topright, 2)


    def _draw_blueprint(self):
        self._draw_arm(self.ARM_BASE_1, self.target_arm1_joints, self.COLOR_BLUEPRINT, None, is_blueprint=True)
        self._draw_arm(self.ARM_BASE_2, self.target_arm2_joints, self.COLOR_BLUEPRINT, None, is_blueprint=True)

    def _draw_arm(self, base_pos, joints, color, glow_color, is_active=False, is_blueprint=False):
        p1, p2 = self._get_arm_endpoints(base_pos, joints)
        
        # Draw glow for active arm
        if is_active and not is_blueprint:
            pygame.gfxdraw.filled_circle(self.screen, int(base_pos[0]), int(base_pos[1]), 18, (*glow_color, 40))
            pygame.gfxdraw.filled_circle(self.screen, int(p1[0]), int(p1[1]), 18, (*glow_color, 40))
            pygame.gfxdraw.aacircle(self.screen, int(base_pos[0]), int(base_pos[1]), 18, (*glow_color, 60))
            pygame.gfxdraw.aacircle(self.screen, int(p1[0]), int(p1[1]), 18, (*glow_color, 60))

        # Draw arm segments
        line_width = 2 if is_blueprint else 6
        pygame.draw.line(self.screen, color, base_pos, p1, line_width)
        pygame.draw.line(self.screen, color, p1, p2, line_width)
        
        # Draw joints
        joint_radius = 6 if is_blueprint else 10
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos[0]), int(base_pos[1]), joint_radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(base_pos[0]), int(base_pos[1]), joint_radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int(p1[0]), int(p1[1]), joint_radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(p1[0]), int(p1[1]), joint_radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int(p2[0]), int(p2[1]), joint_radius-2, color)
        pygame.gfxdraw.aacircle(self.screen, int(p2[0]), int(p2[1]), joint_radius-2, color)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
                radius = int(p['size'] * (p['life'] / p['max_life']))
                if radius > 0:
                     pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]),
                                                  radius, (*p['color'], alpha))

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            life = random.randint(20, 40)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(2, 5)
            })
    
    def _render_text(self, text, pos, font, color, shadow_color=None):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # Score and Widget Count
        self._render_text(f"SCORE: {int(self.score)}", (20, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._render_text(f"WIDGETS: {self.widgets_assembled} / {self.WIDGETS_TO_WIN}", (self.WIDTH - 220, 10), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Active Arm Indicator
        arm_text = "ACTIVE ARM: 1" if self.active_arm == 0 else "ACTIVE ARM: 2"
        arm_color = self.COLOR_ARM1 if self.active_arm == 0 else self.COLOR_ARM2
        self._render_text(arm_text, (20, 45), self.font_small, arm_color)
        self._render_text("Toggle: [SHIFT]", (20, 65), self.font_small, self.COLOR_TEXT)
        
        # Assembly action indicator
        self._render_text("Assemble: [SPACE]", (self.WIDTH - 180, 45), self.font_small, self.COLOR_TEXT)
        
        # Assembly Feedback
        if self.assembly_feedback_timer > 0:
            self.assembly_feedback_timer -= 1
            alpha = int(255 * (self.assembly_feedback_timer / 30))
            if self.last_assembly_status == 'success':
                text, color = "SYNCHRONIZED!", self.COLOR_SUCCESS
            else:
                text, color = "SYNC FAILED!", self.COLOR_FAIL
            
            feedback_surf = self.font_main.render(text, True, color)
            feedback_surf.set_alpha(alpha)
            text_rect = feedback_surf.get_rect(center=(self.WIDTH // 2, 50))
            self.screen.blit(feedback_surf, text_rect)

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block will not run in a headless environment without a display.
    # It's intended for local testing with a GUI.
    # To run, you might need to comment out the SDL_VIDEODRIVER line at the top.
    
    # Check if a display is available before trying to create a window
    if "DISPLAY" in os.environ:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # Mapping keyboard keys to actions
        key_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        # Create a window for human rendering
        pygame.display.set_caption("Robo-Assembler")
        human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

        while not done:
            # Default action is "do nothing"
            action = [0, 0, 0] # [movement, space, shift]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            
            # Check movement keys
            for key, move_action in key_map.items():
                if keys[key]:
                    action[0] = move_action
                    break # Only one movement at a time
            
            # Check modifier keys
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Render the observation to the human-visible screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()

            env.clock.tick(30) # Limit to 30 FPS for human play
        
        print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
        pygame.quit()
    else:
        print("No display found. Skipping manual play example.")
        # Example of headless run
        env = GameEnv()
        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        while not (terminated or truncated) and step_count < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
        print("Headless run finished.")