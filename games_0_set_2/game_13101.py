import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:54:58.795342
# Source Brief: brief_03101.md
# Brief Index: 3101
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a spaceship through a pulsing nebula.
    The goal is to collect energy orbs for speed and reach an exit portal within a time limit,
    while avoiding collision with the nebula walls.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a spaceship through a pulsing nebula, collecting energy orbs to reach the exit portal before time runs out. "
        "Avoid colliding with the dangerous, shifting nebula walls."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to steer your ship through the nebula."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60 # Not used for timing, but for physics tuning
    MAX_STEPS = 120 * FPS # 120 seconds

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_SHIP = (255, 255, 0)
    COLOR_SHIP_GLOW = (255, 200, 0, 50)
    COLOR_ORB = (0, 255, 150)
    COLOR_ORB_GLOW = (100, 255, 200, 80)
    COLOR_NEBULA_OUTER = (80, 40, 180, 100)
    COLOR_NEBULA_INNER = (150, 80, 220, 150)
    COLOR_EXIT = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    # Player Physics
    PLAYER_ACCEL = 0.15
    PLAYER_DRAG = 0.97
    PLAYER_MAX_SPEED = 8.0
    PLAYER_SIZE = 10

    # Nebula Generation
    LEVEL_LENGTH = 20000
    NEBULA_SEGMENT_LENGTH = 40
    NEBULA_BASE_WIDTH = 150 # Half-width (from center to wall)
    NEBULA_AMPLITUDE = 60
    NEBULA_FREQUENCY = 0.005
    NEBULA_PULSE_BASE_FREQ = 0.02
    
    # Orb constants
    NUM_ORBS = 30
    ORB_SIZE = 7

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 18)

        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.player_vel = None
        self.player_trail = None
        
        self.nebula_path_top = None
        self.nebula_path_bottom = None
        self.level_progress = None
        self.pulse_phase = None
        self.pulse_frequency_multiplier = None

        self.orbs = None
        self.orbs_collected = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # This is for dev, not for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_trail = []

        # Game state
        self.steps = 0
        self.score = 0
        self.orbs_collected = 0
        self.game_over = False
        
        # World state
        self.level_progress = 0
        self.pulse_phase = 0
        self.pulse_frequency_multiplier = 1.0
        self._generate_nebula_path()
        self._spawn_orbs(self.NUM_ORBS)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Input & Update Player ---
        self._handle_input(action)
        self._update_player()

        # --- 2. Update World ---
        self._update_world()
        
        # --- 3. Handle Events & Collisions ---
        reward += self._handle_orb_collection()
        collision = self._check_collisions()

        # --- 4. Calculate Reward & Termination ---
        terminated = False
        truncated = False

        if collision:
            reward = -100.0
            self.game_over = True
            terminated = True
            # sfx: explosion
        elif self.level_progress >= self.LEVEL_LENGTH:
            reward = 100.0
            self.game_over = True
            terminated = True
            # sfx: victory fanfare
        elif self.steps >= self.MAX_STEPS:
            # Time limit reached is a truncation, not a failure termination
            truncated = True
            self.game_over = True
            # sfx: failure sound
        else:
            # Survival and progress reward
            reward += 0.01 + (self.player_vel.x * 0.001)

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        if movement == 1: # Up
            self.player_vel.y -= self.PLAYER_ACCEL
        if movement == 2: # Down
            self.player_vel.y += self.PLAYER_ACCEL
        if movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        if movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL

    def _update_player(self):
        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Clamp speed
        speed = self.player_vel.length()
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        if speed < -self.PLAYER_MAX_SPEED:
             self.player_vel.scale_to_length(-self.PLAYER_MAX_SPEED)

        # Update position
        self.player_pos += self.player_vel

        # Clamp position to screen bounds
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_SIZE))
        self.player_pos.y = max(self.PLAYER_SIZE, min(self.player_pos.y, self.SCREEN_HEIGHT - self.PLAYER_SIZE))

        # Update trail
        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > 15:
            self.player_trail.pop(0)

    def _update_world(self):
        # Scroll the world based on player's forward velocity
        self.level_progress += self.player_vel.x
        self.pulse_phase += self.NEBULA_PULSE_BASE_FREQ * self.pulse_frequency_multiplier
        
        # Increase pulse rate every 30 seconds
        if self.steps > 0 and self.steps % (30 * self.FPS) == 0:
            self.pulse_frequency_multiplier += 0.05
    
    def _generate_nebula_path(self):
        self.nebula_path_top = []
        self.nebula_path_bottom = []
        num_segments = math.ceil(self.LEVEL_LENGTH / self.NEBULA_SEGMENT_LENGTH) + math.ceil(self.SCREEN_WIDTH / self.NEBULA_SEGMENT_LENGTH)
        
        # Use multiple sine waves for a more organic look
        freq1, amp1 = self.NEBULA_FREQUENCY * self.np_random.uniform(0.8, 1.2), self.NEBULA_AMPLITUDE * self.np_random.uniform(0.7, 1.3)
        freq2, amp2 = self.NEBULA_FREQUENCY * 2.5 * self.np_random.uniform(0.8, 1.2), self.NEBULA_AMPLITUDE * 0.4 * self.np_random.uniform(0.7, 1.3)
        phase1, phase2 = self.np_random.uniform(0, 2 * math.pi), self.np_random.uniform(0, 2 * math.pi)

        for i in range(num_segments):
            x = i * self.NEBULA_SEGMENT_LENGTH
            offset = amp1 * math.sin(x * freq1 + phase1) + amp2 * math.sin(x * freq2 + phase2)
            
            y_top = self.SCREEN_HEIGHT / 2 - self.NEBULA_BASE_WIDTH + offset
            y_bottom = self.SCREEN_HEIGHT / 2 + self.NEBULA_BASE_WIDTH + offset
            
            self.nebula_path_top.append(y_top)
            self.nebula_path_bottom.append(y_bottom)

    def _spawn_orbs(self, count):
        self.orbs = []
        for _ in range(count):
            progress = self.np_random.uniform(500, self.LEVEL_LENGTH - 500)
            
            # Find nebula walls at this progress point
            idx = int(progress / self.NEBULA_SEGMENT_LENGTH)
            if idx >= len(self.nebula_path_top):
                continue
            y_top = self.nebula_path_top[idx]
            y_bottom = self.nebula_path_bottom[idx]
            
            # Spawn orb safely inside the corridor
            y = self.np_random.uniform(y_top + self.ORB_SIZE * 4, y_bottom - self.ORB_SIZE * 4)
            self.orbs.append(pygame.Vector2(progress, y))

    def _handle_orb_collection(self):
        reward = 0
        orbs_to_remove = []
        for orb in self.orbs:
            # Orb position relative to screen
            orb_screen_x = orb.x - self.level_progress + self.player_pos.x
            orb_screen_pos = pygame.Vector2(orb_screen_x, orb.y)
            
            if orb_screen_pos.distance_to(self.player_pos) < self.PLAYER_SIZE + self.ORB_SIZE:
                orbs_to_remove.append(orb)
                self.score += 1.0
                reward += 1.0
                self.orbs_collected += 1
                # sfx: orb collection
        
        for orb in orbs_to_remove:
            self.orbs.remove(orb)
            
        return reward
        
    def _check_collisions(self):
        # Check collision with nebula walls
        # Find the segment of the nebula path the player is currently in
        player_world_x = self.level_progress
        path_idx = int(player_world_x / self.NEBULA_SEGMENT_LENGTH)
        
        if path_idx < 0 or path_idx >= len(self.nebula_path_top) -1:
            return True # Player is off the path

        # Pulsing effect
        pulse_amount = (math.sin(self.pulse_phase) + 1) / 2 * (self.NEBULA_BASE_WIDTH * 0.4)
        
        # Interpolate between path points for smooth walls
        ratio = (player_world_x % self.NEBULA_SEGMENT_LENGTH) / self.NEBULA_SEGMENT_LENGTH
        
        wall_top_y = self.nebula_path_top[path_idx] * (1 - ratio) + self.nebula_path_top[path_idx + 1] * ratio
        wall_bottom_y = self.nebula_path_bottom[path_idx] * (1 - ratio) + self.nebula_path_bottom[path_idx + 1] * ratio

        # Check collision
        if self.player_pos.y - self.PLAYER_SIZE < wall_top_y + pulse_amount:
            return True
        if self.player_pos.y + self.PLAYER_SIZE > wall_bottom_y - pulse_amount:
            return True
            
        return False

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
            "orbs_collected": self.orbs_collected,
            "level_progress": self.level_progress,
            "player_speed": self.player_vel.length()
        }

    def _render_game(self):
        # --- Render Nebula ---
        start_idx = max(0, int((self.level_progress - self.player_pos.x) / self.NEBULA_SEGMENT_LENGTH))
        end_idx = min(len(self.nebula_path_top), start_idx + int(self.SCREEN_WIDTH / self.NEBULA_SEGMENT_LENGTH) + 3)
        pulse_amount = (math.sin(self.pulse_phase) + 1) / 2 * (self.NEBULA_BASE_WIDTH * 0.4)
        
        for i in range(2): # Draw two layers for depth
            layer_pulse = pulse_amount * (1.5 - i * 0.5)
            layer_color = self.COLOR_NEBULA_OUTER if i == 0 else self.COLOR_NEBULA_INNER
            
            # Top wall
            points_top = []
            for j in range(start_idx, end_idx):
                x = j * self.NEBULA_SEGMENT_LENGTH - self.level_progress + self.player_pos.x
                y = self.nebula_path_top[j] + layer_pulse
                points_top.append((int(x), int(y)))
            if len(points_top) > 1:
                poly_points_top = [(points_top[0][0], -10)] + points_top + [(points_top[-1][0], -10)]
                pygame.gfxdraw.filled_polygon(self.screen, poly_points_top, layer_color)

            # Bottom wall
            points_bottom = []
            for j in range(start_idx, end_idx):
                x = j * self.NEBULA_SEGMENT_LENGTH - self.level_progress + self.player_pos.x
                y = self.nebula_path_bottom[j] - layer_pulse
                points_bottom.append((int(x), int(y)))
            if len(points_bottom) > 1:
                poly_points_bottom = [(points_bottom[0][0], self.SCREEN_HEIGHT + 10)] + points_bottom + [(points_bottom[-1][0], self.SCREEN_HEIGHT + 10)]
                pygame.gfxdraw.filled_polygon(self.screen, poly_points_bottom, layer_color)

        # --- Render Orbs ---
        for orb in self.orbs:
            orb_screen_x = orb.x - self.level_progress + self.player_pos.x
            if 0 < orb_screen_x < self.SCREEN_WIDTH:
                pos = (int(orb_screen_x), int(orb.y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ORB_SIZE + 3, self.COLOR_ORB_GLOW)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ORB_SIZE + 3, self.COLOR_ORB_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ORB_SIZE, self.COLOR_ORB)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ORB_SIZE, self.COLOR_ORB)
        
        # --- Render Exit Portal ---
        exit_screen_x = self.LEVEL_LENGTH - self.level_progress + self.player_pos.x
        if exit_screen_x < self.SCREEN_WIDTH + 100:
            for i in range(20, 0, -2):
                alpha = 255 - i * 12
                pygame.gfxdraw.aacircle(self.screen, int(exit_screen_x), self.SCREEN_HEIGHT // 2, 10 + i, (*self.COLOR_EXIT, alpha))

        # --- Render Player Trail ---
        if self.player_trail:
            for i, p in enumerate(self.player_trail):
                alpha = int(255 * (i / len(self.player_trail)))
                size = int(self.PLAYER_SIZE * 0.5 * (i / len(self.player_trail)))
                if size > 0:
                    trail_color = (*self.COLOR_SHIP[:3], alpha)
                    pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), size, trail_color)

        # --- Render Player ---
        px, py = int(self.player_pos.x), int(self.player_pos.y)
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_SIZE * 2, self.COLOR_SHIP_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_SIZE * 2, self.COLOR_SHIP_GLOW)
        # Ship body
        angle = self.player_vel.angle_to(pygame.Vector2(1, 0))
        p1 = (px + math.cos(math.radians(angle)) * self.PLAYER_SIZE, py - math.sin(math.radians(angle)) * self.PLAYER_SIZE)
        p2 = (px + math.cos(math.radians(angle - 140)) * self.PLAYER_SIZE, py - math.sin(math.radians(angle - 140)) * self.PLAYER_SIZE)
        p3 = (px + math.cos(math.radians(angle + 140)) * self.PLAYER_SIZE, py - math.sin(math.radians(angle + 140)) * self.PLAYER_SIZE)
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_SHIP)

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (10, 10))

        # Speed
        speed = self.player_vel.length()
        speed_text = self.font_main.render(f"SPEED: {speed:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 10, 10))

        # Orbs Collected
        orbs_text = self.font_small.render(f"ORBS: {self.orbs_collected}", True, self.COLOR_TEXT)
        self.screen.blit(orbs_text, (10, self.SCREEN_HEIGHT - orbs_text.get_height() - 10))
        
        # Progress Bar
        progress_ratio = self.level_progress / self.LEVEL_LENGTH
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 5
        bar_x = 10
        bar_y = self.SCREEN_HEIGHT - bar_height - 2
        pygame.draw.rect(self.screen, (255, 255, 255, 50), (bar_x, bar_y, bar_width, bar_height), 1)
        pygame.draw.rect(self.screen, self.COLOR_ORB, (bar_x, bar_y, int(bar_width * progress_ratio), bar_height))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # This block will not run with the dummy video driver unless a display is available.
    # To run, you might need to unset the SDL_VIDEODRIVER environment variable.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # We can't use env.render() as it's not defined for 'human' mode in the brief.
    # So we'll create a window here just for the demo.
    
    try:
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Nebula Runner")
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        
        print("\n--- Manual Control ---")
        print("Arrows: Move")
        print("Q: Quit")

        while not done:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    done = True
            
            if done:
                break

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode Finished. Final Score: {info['score']:.2f}, Progress: {info['level_progress']/env.LEVEL_LENGTH:.1%}")
                obs, info = env.reset()

            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(GameEnv.FPS)

    finally:
        env.close()