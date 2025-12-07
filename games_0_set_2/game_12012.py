import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:52:33.866017
# Source Brief: brief_02012.md
# Brief Index: 2012
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must balance a seesaw.

    The agent controls the weights on both ends of the seesaw to counteract
    random fluctuations. The goal is to keep the seesaw balanced for 60 seconds.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement):
        - 0: No-op
        - 1: Increase left weight
        - 2: Decrease left weight
        - 3: Decrease right weight
        - 4: Increase right weight
    - `action[1]` (Space): Increase both weights
    - `action[2]` (Shift): Decrease both weights

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Reward Structure:**
    - +0.1 per step for being balanced (within a small angle threshold).
    - -0.1 per step for being tilted.
    - +100 bonus for surviving the full duration.

    **Termination:**
    - The episode ends if the score drops below -10.
    - The episode ends after 6000 steps (60 seconds).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance a seesaw by adjusting weights on either side to counteract random forces. "
        "Survive for 60 seconds to win."
    )
    user_guide = (
        "Controls: Use ↑/↓ to adjust the left weight and ←/→ to adjust the right weight. "
        "Press space to increase both weights and shift to decrease both."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 60 # Note: step is called per frame, not per second.
    MAX_STEPS = 3600 # 60 seconds * 60 FPS

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_PLANK = (110, 80, 60)
    COLOR_PLANK_OUTLINE = (70, 50, 40)
    COLOR_FULCRUM = (100, 105, 110)
    COLOR_FULCRUM_OUTLINE = (60, 65, 70)
    COLOR_LEFT_WEIGHT = (255, 70, 70)
    COLOR_RIGHT_WEIGHT = (70, 150, 255)
    COLOR_TEXT = (230, 230, 240)
    COLOR_BALANCED = (100, 255, 100)
    COLOR_HEALTH_BAR_FULL = (60, 200, 60)
    COLOR_HEALTH_BAR_EMPTY = (200, 60, 60)
    
    # Physics
    PLANK_LENGTH = 450
    PLANK_THICKNESS = 18
    FULCRUM_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
    TORQUE_FACTOR = 0.0003
    ANGULAR_DAMPING = 0.96
    MAX_ANGLE_DEGREES = 35.0

    # Gameplay
    INITIAL_WEIGHT = 5.0
    MIN_WEIGHT = 1.0
    MAX_WEIGHT = 15.0
    WEIGHT_ADJUST_RATE = 0.2
    BOTH_ADJUST_RATE = 0.1
    INITIAL_FLUCTUATION = 0.2
    DIFFICULTY_INCREASE_INTERVAL = 600 # Every 10 seconds (600 steps / 60 FPS)
    DIFFICULTY_INCREASE_AMOUNT = 0.1
    
    # Rewards & Termination
    SCORE_LOSS_THRESHOLD = -10.0
    BALANCED_ANGLE_DEGREES = 3.0
    REWARD_BALANCE = 0.1
    REWARD_TILT = -0.1
    REWARD_WIN = 100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 18)

        # Pre-create plank surface for rotation
        self.plank_surface = pygame.Surface((self.PLANK_LENGTH, self.PLANK_THICKNESS), pygame.SRCALPHA)
        pygame.draw.rect(self.plank_surface, self.COLOR_PLANK, self.plank_surface.get_rect(), border_radius=5)
        pygame.draw.rect(self.plank_surface, self.COLOR_PLANK_OUTLINE, self.plank_surface.get_rect(), width=3, border_radius=5)

        # Initialize state variables
        self.steps = 0
        self.score = 0.0
        self.left_weight = self.INITIAL_WEIGHT
        self.right_weight = self.INITIAL_WEIGHT
        self.seesaw_angle_rad = 0.0
        self.angular_velocity = 0.0
        self.fluctuation_magnitude = 0.0
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        
        self.left_weight = self.INITIAL_WEIGHT
        self.right_weight = self.INITIAL_WEIGHT
        
        self.seesaw_angle_rad = 0.0
        self.angular_velocity = 0.0
        
        self.fluctuation_magnitude = self.INITIAL_FLUCTUATION
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        # 1. Handle actions and update weights
        self._handle_actions(action)
        
        # 2. Update physics
        self._update_physics()

        # 3. Update game logic
        self.steps += 1
        self._update_difficulty()
        
        # 4. Update particle effects
        self._update_particles()
        
        # 5. Calculate reward and check for termination
        terminated = False
        reward = 0

        if abs(math.degrees(self.seesaw_angle_rad)) <= self.BALANCED_ANGLE_DEGREES:
            reward = self.REWARD_BALANCE
        else:
            reward = self.REWARD_TILT
        
        self.score += reward
        
        if self.score <= self.SCORE_LOSS_THRESHOLD:
            terminated = True
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limits
            if self.score > self.SCORE_LOSS_THRESHOLD:
                reward += self.REWARD_WIN # Win bonus
                self.score += self.REWARD_WIN

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        old_left_weight = self.left_weight
        old_right_weight = self.right_weight

        # Movement actions
        if movement == 1: self.left_weight += self.WEIGHT_ADJUST_RATE
        elif movement == 2: self.left_weight -= self.WEIGHT_ADJUST_RATE
        elif movement == 3: self.right_weight -= self.WEIGHT_ADJUST_RATE
        elif movement == 4: self.right_weight += self.WEIGHT_ADJUST_RATE

        # Simultaneous button actions
        if space_held:
            self.left_weight += self.BOTH_ADJUST_RATE
            self.right_weight += self.BOTH_ADJUST_RATE
        if shift_held:
            self.left_weight -= self.BOTH_ADJUST_RATE
            self.right_weight -= self.BOTH_ADJUST_RATE

        # Clamp weights
        self.left_weight = np.clip(self.left_weight, self.MIN_WEIGHT, self.MAX_WEIGHT)
        self.right_weight = np.clip(self.right_weight, self.MIN_WEIGHT, self.MAX_WEIGHT)

        # Spawn particles on weight change
        if self.left_weight != old_left_weight:
            self._spawn_particles(side='left', change=self.left_weight - old_left_weight)
        if self.right_weight != old_right_weight:
            self._spawn_particles(side='right', change=self.right_weight - old_right_weight)

    def _update_physics(self):
        fluctuation = self.np_random.uniform(-self.fluctuation_magnitude, self.fluctuation_magnitude)
        
        # Torque is proportional to weight difference plus fluctuation
        torque = (self.right_weight - self.left_weight) * self.TORQUE_FACTOR + fluctuation
        
        self.angular_velocity += torque
        self.angular_velocity *= self.ANGULAR_DAMPING
        self.seesaw_angle_rad += self.angular_velocity
        
        # Clamp angle
        max_angle_rad = math.radians(self.MAX_ANGLE_DEGREES)
        self.seesaw_angle_rad = np.clip(self.seesaw_angle_rad, -max_angle_rad, max_angle_rad)
        
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DIFFICULTY_INCREASE_INTERVAL == 0:
            self.fluctuation_magnitude += self.DIFFICULTY_INCREASE_AMOUNT

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "left_weight": self.left_weight,
            "right_weight": self.right_weight,
            "angle_degrees": math.degrees(self.seesaw_angle_rad),
        }

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, p['life'] / p['max_life'])
            color = (*p['color'], int(alpha * 255))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

        # Render fulcrum
        fulcrum_points = [
            (self.FULCRUM_POS[0], self.FULCRUM_POS[1] - 25),
            (self.FULCRUM_POS[0] - 40, self.FULCRUM_POS[1] + 25),
            (self.FULCRUM_POS[0] + 40, self.FULCRUM_POS[1] + 25)
        ]
        pygame.gfxdraw.aapolygon(self.screen, fulcrum_points, self.COLOR_FULCRUM_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, fulcrum_points, self.COLOR_FULCRUM)
        
        # Render plank
        angle_deg = -math.degrees(self.seesaw_angle_rad)
        rotated_plank = pygame.transform.rotate(self.plank_surface, angle_deg)
        plank_rect = rotated_plank.get_rect(center=self.FULCRUM_POS)
        self.screen.blit(rotated_plank, plank_rect)

        # Render weights
        self._render_weight('left')
        self._render_weight('right')

    def _render_weight(self, side):
        arm = self.PLANK_LENGTH / 2 - 25
        angle = self.seesaw_angle_rad
        
        if side == 'left':
            weight = self.left_weight
            color = self.COLOR_LEFT_WEIGHT
            pos_x = self.FULCRUM_POS[0] - arm * math.cos(angle)
            pos_y = self.FULCRUM_POS[1] - arm * math.sin(angle)
        else: # right
            weight = self.right_weight
            color = self.COLOR_RIGHT_WEIGHT
            pos_x = self.FULCRUM_POS[0] + arm * math.cos(angle)
            pos_y = self.FULCRUM_POS[1] + arm * math.sin(angle)
            
        # Size is function of weight
        size = int(10 + weight * 1.5)
        
        # Position the weight on top of the plank
        offset_x = (self.PLANK_THICKNESS / 2 + size) * math.sin(angle)
        offset_y = (self.PLANK_THICKNESS / 2 + size) * math.cos(angle)
        
        final_x = int(pos_x - offset_x)
        final_y = int(pos_y - offset_y)
        
        # Draw glow effect
        for i in range(4):
            glow_alpha = 40 - i * 10
            glow_size = size + i * 3
            glow_color = (*color, glow_alpha)
            pygame.gfxdraw.filled_circle(self.screen, final_x, final_y, glow_size, glow_color)
        
        # Draw main weight circle
        pygame.gfxdraw.aacircle(self.screen, final_x, final_y, size, color)
        pygame.gfxdraw.filled_circle(self.screen, final_x, final_y, size, color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Health Bar (visualizing score proximity to loss)
        health_percent = max(0, (self.score - self.SCORE_LOSS_THRESHOLD) / abs(self.SCORE_LOSS_THRESHOLD))
        bar_width = 200
        bar_height = 10
        bar_x = 10
        bar_y = 40
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_EMPTY, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        if health_percent > 0:
            fill_width = int(bar_width * health_percent)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FULL, (bar_x, bar_y, fill_width, bar_height), border_radius=3)

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.TARGET_FPS
        timer_text = self.font_timer.render(f"TIME: {time_left:.1f}s", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Balance indicator
        is_balanced = abs(math.degrees(self.seesaw_angle_rad)) <= self.BALANCED_ANGLE_DEGREES
        balance_color = self.COLOR_BALANCED if is_balanced else self.COLOR_LEFT_WEIGHT
        balance_text = "BALANCED" if is_balanced else "TILTED"
        balance_surf = self.font_timer.render(balance_text, True, balance_color)
        balance_rect = balance_surf.get_rect(midtop=(self.SCREEN_WIDTH // 2, 10))
        self.screen.blit(balance_surf, balance_rect)

    def _spawn_particles(self, side, change):
        # Determine particle properties based on weight change
        num_particles = min(15, int(abs(change) * 20))
        color = (100, 255, 100) if change > 0 else (255, 100, 100) # Green for add, Red for remove
        
        # Calculate spawn position
        arm = self.PLANK_LENGTH / 2 - 25
        angle = self.seesaw_angle_rad
        if side == 'left':
            pos_x = self.FULCRUM_POS[0] - arm * math.cos(angle)
            pos_y = self.FULCRUM_POS[1] - arm * math.sin(angle)
        else: # right
            pos_x = self.FULCRUM_POS[0] + arm * math.cos(angle)
            pos_y = self.FULCRUM_POS[1] + arm * math.sin(angle)

        for _ in range(num_particles):
            # Sound effect placeholder: pygame.mixer.Sound('pop.wav').play()
            angle_rad = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle_rad) * speed, math.sin(angle_rad) * speed]
            life = self.np_random.integers(20, 40)
            size = self.np_random.uniform(2, 5)
            self.particles.append({
                'pos': [pos_x, pos_y],
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': size,
                'color': color
            })

    def _update_particles(self):
        # Use a list comprehension to filter out dead particles and update live ones
        new_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][1] += 0.1 # Gravity
                p['size'] *= 0.98 # Shrink
                new_particles.append(p)
        self.particles = new_particles

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Seesaw Balance")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Controls ---")
    print("W/S:      Adjust Left Weight (Increase/Decrease)")
    print("A/D:      Adjust Right Weight (Decrease/Increase)")
    print("SPACE:    Increase Both Weights")
    print("L. SHIFT: Decrease Both Weights")
    print("R:        Reset Environment")
    print("Q:        Quit")
    print("----------------\n")
    print("Note: The W/A/S/D mapping is for manual testing. An agent would use actions 0-4.")
    print("W -> Action 1, S -> Action 2, A -> Action 3, D -> Action 4")


    while running:
        # --- Human Input to Action Mapping ---
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: movement = 1   # Increase left
        elif keys[pygame.K_s]: movement = 2 # Decrease left
        elif keys[pygame.K_a]: movement = 3 # Decrease right
        elif keys[pygame.K_d]: movement = 4 # Increase right
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
                    total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already the rendered image
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.TARGET_FPS)

    env.close()