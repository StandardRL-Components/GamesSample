import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:55:51.125276
# Source Brief: brief_00164.md
# Brief Index: 164
# """import gymnasium as gym

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manipulates time and gravity 
    to fling seeds into fertile patches of a meadow.

    The goal is to plant a target number of seeds within a time limit.
    The primary challenge lies in mastering the physics-based trajectory,
    which is affected by changing wind (gravity) and player-controlled
    time dilation.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Fling seeds into fertile patches of a meadow by mastering physics-based trajectories. "
        "Manipulate time and account for changing winds to plant all your seeds before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ to aim the launcher and ↑↓ to adjust power. "
        "Hold space to slow time and aim, then release to launch a seed. "
        "Press shift to cycle between seed types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60 # Internal simulation FPS, not necessarily render FPS
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 60)
    COLOR_BARREN_GROUND = (50, 30, 20)
    COLOR_FERTILE_GROUND = (40, 80, 40)
    COLOR_LAUNCHER = (220, 220, 240)
    COLOR_LAUNCHER_ACCENT = (180, 180, 200)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_AURA = (100, 150, 255)
    COLOR_WIND_INDICATOR = (150, 200, 255, 150)
    
    # Game Parameters
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * 10 # Assuming 10 steps per second for RL agent
    
    LAUNCHER_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)
    MIN_ANGLE = -160
    MAX_ANGLE = -20
    MIN_POWER = 2.0
    MAX_POWER = 10.0
    MIN_TIME_DILATION = 0.1
    MAX_TIME_DILATION = 1.0
    
    # Seed Types
    SEED_DEFS = {
        'standard': {'color': (255, 220, 100), 'gravity_mult': 1.0, 'radius': 4},
        'heavy': {'color': (255, 150, 50), 'gravity_mult': 1.5, 'radius': 5},
        'light': {'color': (200, 255, 255), 'gravity_mult': 0.6, 'radius': 3},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Persistent State (survives resets)
        self.unlocked_seeds = ['standard']
        self.difficulty_level = 0
        
        # Initialize state variables that are reset each episode
        self._initialize_state()

        # Validate implementation
        # self.validate_implementation() # Uncomment to run validation check

    def _initialize_state(self):
        """Initializes all non-persistent state variables."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.game_timer = self.MAX_STEPS
        self.target_score = 10 + self.difficulty_level * 5

        # Player state
        self.launcher_angle = -90.0
        self.launcher_power = (self.MIN_POWER + self.MAX_POWER) / 2
        self.time_dilation = self.MAX_TIME_DILATION
        
        # Action state tracking
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Game objects
        self.seeds = []
        self.particles = []
        self.fertile_patches = []
        
        # Physics
        self.gravity = pygame.Vector2(0, 0.1)
        self.wind_change_interval = max(50, 150 - self.difficulty_level * 10)
        self.wind_timer = self.wind_change_interval

        # Seed selection
        self.current_seed_index = 0
        
        # Reward buffer for the current step
        self.reward_buffer = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_state()
        
        # Procedurally generate fertile patches
        num_patches = self.np_random.integers(5, 10)
        for _ in range(num_patches):
            width = self.np_random.integers(50, 150)
            height = self.np_random.integers(10, 30)
            x = self.np_random.integers(0, self.SCREEN_WIDTH - width)
            y = self.np_random.integers(self.SCREEN_HEIGHT // 2, self.SCREEN_HEIGHT - 80)
            self.fertile_patches.append(pygame.Rect(x, y, width, height))

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_buffer = -0.01 # Small penalty for existing
        
        self._handle_input(action)
        self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()
        
        reward = self.reward_buffer
        if terminated:
            if self.score >= self.target_score:
                reward += 50.0 # Big reward for winning
                self.difficulty_level += 1 # Increase difficulty for next game
                self._unlock_seeds()
            else:
                reward -= 10.0 # Penalty for losing
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Launcher Rotation (Left/Right) ---
        if movement == 3: # Left
            self.launcher_angle -= 1.5
        elif movement == 4: # Right
            self.launcher_angle += 1.5
        self.launcher_angle = np.clip(self.launcher_angle, self.MIN_ANGLE, self.MAX_ANGLE)

        # --- Launcher Power (Up/Down) ---
        if movement == 1: # Up
            self.launcher_power += 0.1
        elif movement == 2: # Down
            self.launcher_power -= 0.1
        self.launcher_power = np.clip(self.launcher_power, self.MIN_POWER, self.MAX_POWER)
        
        # --- Time Dilation ---
        if space_held:
            self.time_dilation -= 0.05
        if shift_held:
            self.time_dilation += 0.05
        if not space_held and not shift_held:
            # Drift back to normal time
            self.time_dilation += 0.02
        self.time_dilation = np.clip(self.time_dilation, self.MIN_TIME_DILATION, self.MAX_TIME_DILATION)

        # --- Launch Seed (on Space release) ---
        if self.prev_space_held and not space_held:
            self._launch_seed()
            # sfx: launch_sound()

        # --- Cycle Seed Type (on Shift release) ---
        if self.prev_shift_held and not shift_held:
            self.current_seed_index = (self.current_seed_index + 1) % len(self.unlocked_seeds)
            # sfx: cycle_weapon_sound()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        # --- Timers ---
        self.game_timer -= 1
        self.wind_timer -= 1

        # --- Wind Change ---
        if self.wind_timer <= 0:
            self.wind_timer = self.wind_change_interval
            wind_strength = self.np_random.uniform(-0.05, 0.05)
            self.gravity.x = wind_strength
            # sfx: wind_change_sound()

        # --- Update Seeds ---
        for seed in self.seeds[:]:
            seed_def = self.SEED_DEFS[seed['type']]
            
            # Apply physics
            seed['vel'] += self.gravity * seed_def['gravity_mult'] * self.time_dilation
            seed['pos'] += seed['vel'] * self.time_dilation
            seed['trail'].append(seed['pos'].copy())
            if len(seed['trail']) > 20:
                seed['trail'].pop(0)

            # Check for landing
            landed = False
            for patch in self.fertile_patches:
                if patch.collidepoint(seed['pos']):
                    self.score += 1
                    self.reward_buffer += 1.0
                    self._create_impact_particles(seed['pos'], self.COLOR_FERTILE_GROUND)
                    landed = True
                    # sfx: plant_success_sound()
                    break
            
            # Check for out of bounds or landing on barren ground
            if landed or not (0 < seed['pos'].x < self.SCREEN_WIDTH and seed['pos'].y < self.SCREEN_HEIGHT):
                if not landed and seed['pos'].y >= self.SCREEN_HEIGHT - 50:
                    self._create_impact_particles(seed['pos'], self.COLOR_BARREN_GROUND)
                    # sfx: plant_fail_sound()
                self.seeds.remove(seed)

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _launch_seed(self):
        angle_rad = math.radians(self.launcher_angle)
        power = self.launcher_power
        
        vel = pygame.Vector2(
            math.cos(angle_rad) * power,
            math.sin(angle_rad) * power
        )
        
        seed_type = self.unlocked_seeds[self.current_seed_index]
        
        self.seeds.append({
            'pos': pygame.Vector2(self.LAUNCHER_POS),
            'vel': vel,
            'type': seed_type,
            'trail': []
        })

    def _create_impact_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _unlock_seeds(self):
        if self.score >= 25 and 'heavy' not in self.unlocked_seeds:
            self.unlocked_seeds.append('heavy')
        if self.score >= 50 and 'light' not in self.unlocked_seeds:
            self.unlocked_seeds.append('light')
            
    def _check_termination(self):
        if self.game_timer <= 0 or self.score >= self.target_score:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_BARREN_GROUND, (0, self.SCREEN_HEIGHT - 50, self.SCREEN_WIDTH, 50))
        
        # Fertile Patches
        for patch in self.fertile_patches:
            pygame.draw.rect(self.screen, self.COLOR_FERTILE_GROUND, patch)

    def _render_game_elements(self):
        # Time Dilation Aura
        if self.time_dilation < 0.95:
            aura_alpha = int(100 * (1 - (self.time_dilation / self.MAX_TIME_DILATION)**2))
            aura_radius = int(50 + 30 * math.sin(self.steps * 0.2))
            if aura_radius > 0:
                s = pygame.Surface((aura_radius * 2, aura_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_AURA, aura_alpha), (aura_radius, aura_radius), aura_radius)
                self.screen.blit(s, (self.LAUNCHER_POS[0] - aura_radius, self.LAUNCHER_POS[1] - aura_radius))

        # Predicted Trajectory
        self._draw_predicted_trajectory()

        # Seeds and Trails
        for seed in self.seeds:
            seed_def = self.SEED_DEFS[seed['type']]
            # Trail
            if len(seed['trail']) > 1:
                points = [(int(p.x), int(p.y)) for p in seed['trail']]
                pygame.draw.aalines(self.screen, seed_def['color'], False, points, 1)
            # Seed
            pygame.gfxdraw.aacircle(self.screen, int(seed['pos'].x), int(seed['pos'].y), seed_def['radius'], seed_def['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(seed['pos'].x), int(seed['pos'].y), seed_def['radius'], seed_def['color'])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # Launcher
        angle_rad = math.radians(self.launcher_angle)
        end_x = self.LAUNCHER_POS[0] + math.cos(angle_rad) * 30
        end_y = self.LAUNCHER_POS[1] + math.sin(angle_rad) * 30
        pygame.draw.aaline(self.screen, self.COLOR_LAUNCHER, self.LAUNCHER_POS, (end_x, end_y), 3)
        pygame.gfxdraw.aacircle(self.screen, self.LAUNCHER_POS[0], self.LAUNCHER_POS[1], 10, self.COLOR_LAUNCHER)
        pygame.gfxdraw.filled_circle(self.screen, self.LAUNCHER_POS[0], self.LAUNCHER_POS[1], 10, self.COLOR_LAUNCHER)
        pygame.gfxdraw.aacircle(self.screen, self.LAUNCHER_POS[0], self.LAUNCHER_POS[1], 6, self.COLOR_LAUNCHER_ACCENT)
        pygame.gfxdraw.filled_circle(self.screen, self.LAUNCHER_POS[0], self.LAUNCHER_POS[1], 6, self.COLOR_LAUNCHER_ACCENT)

    def _draw_predicted_trajectory(self):
        angle_rad = math.radians(self.launcher_angle)
        vel = pygame.Vector2(math.cos(angle_rad) * self.launcher_power, math.sin(angle_rad) * self.launcher_power)
        pos = pygame.Vector2(self.LAUNCHER_POS)
        
        seed_type = self.unlocked_seeds[self.current_seed_index]
        seed_def = self.SEED_DEFS[seed_type]
        gravity_mult = seed_def['gravity_mult']
        
        points = []
        for i in range(100): # 100 steps of prediction
            vel += self.gravity * gravity_mult
            pos += vel
            if i % 5 == 0: # Draw a point every 5 steps
                points.append((int(pos.x), int(pos.y)))
        
        if len(points) > 1:
            # Draw dashed line
            for i in range(len(points) - 1):
                if i % 2 == 0:
                    pygame.draw.aaline(self.screen, (255, 255, 255, 100), points[i], points[i+1])

    def _render_ui(self):
        # Score and Target
        score_text = self.font_large.render(f"PLANTED: {self.score}/{self.target_score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = max(0, self.game_timer // 10)
        timer_text = self.font_large.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # Wind Indicator
        wind_strength = self.gravity.x
        arrow_start = (self.SCREEN_WIDTH // 2, 25)
        arrow_end = (arrow_start[0] + wind_strength * 300, 25)
        pygame.draw.line(self.screen, self.COLOR_WIND_INDICATOR, arrow_start, arrow_end, 2)
        if abs(wind_strength) > 0.005:
            p1 = (arrow_end[0] - 5 * np.sign(wind_strength), arrow_end[1] - 5)
            p2 = (arrow_end[0] - 5 * np.sign(wind_strength), arrow_end[1] + 5)
            pygame.draw.polygon(self.screen, self.COLOR_WIND_INDICATOR, [arrow_end, p1, p2])

        # Launcher Info
        power_percent = (self.launcher_power - self.MIN_POWER) / (self.MAX_POWER - self.MIN_POWER)
        
        # Power Bar
        bar_width = 100
        bar_height = 10
        bar_x = self.LAUNCHER_POS[0] - bar_width // 2
        bar_y = self.SCREEN_HEIGHT - 15
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (255,100,100), (bar_x, bar_y, bar_width * power_percent, bar_height))
        
        # Seed Type
        seed_type_name = self.unlocked_seeds[self.current_seed_index].upper()
        seed_color = self.SEED_DEFS[seed_type_name.lower()]['color']
        seed_text = self.font_small.render(f"SEED: {seed_type_name}", True, seed_color)
        self.screen.blit(seed_text, (bar_x + bar_width + 10, bar_y - 2))

    def _get_info(self):
        return {
            "score": self.score,
            "target_score": self.target_score,
            "steps": self.steps,
            "game_timer": self.game_timer,
            "difficulty_level": self.difficulty_level,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("--- Running Implementation Validation ---")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        print("✓ Action space validated.")
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        print("✓ Observation space validated.")
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        print("✓ reset() method validated.")
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ step() method validated.")
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment visually
if __name__ == '__main__':
    # This block will not run with the SDL_VIDEODRIVER="dummy" set by default.
    # To run visually, comment out the os.environ line at the top of the file.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run example visually with SDL_VIDEODRIVER='dummy'.")
        print("Comment out the os.environ.setdefault line to run this example.")
        exit()

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Seed Flinger Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default no-op action
        
        # Manual keyboard controls for testing
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: action[0] = 1 # Power up
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: action[0] = 2 # Power down
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: action[0] = 3 # Rotate left
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: action[0] = 4 # Rotate right
        if keys[pygame.K_SPACE]: action[1] = 1 # Time Dilation
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1 # Cycle seed
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Environment Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode Finished! Final Score: {info['score']}/{info['target_score']}. Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit frame rate for human playability
        
    env.close()