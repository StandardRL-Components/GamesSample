import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:17:05.756863
# Source Brief: brief_00307.md
# Brief Index: 307
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the goal is to control four conveyor belts
    to synchronize the flow of colored cubes. The player adjusts the speed of
    three belts to match the color of a fourth reference belt at a target zone.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the speed of three conveyor belts to synchronize the colors of cubes "
        "passing through a target zone, matching them to a fourth reference belt."
    )
    user_guide = (
        "Controls: Use ↑/↓ to adjust Belt 1 speed, ←/→ for Belt 2, and space/shift for Belt 3. "
        "Match all cube colors in the yellow target zone to win."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    DT = 1.0 / FPS

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (30, 40, 50)
    COLOR_BELT = (45, 55, 65)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_TARGET_LINE = (255, 255, 0)
    CUBE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
    ]

    # Game Parameters
    BELT_COUNT = 4
    BELT_HEIGHT = 40
    BELT_SPACING = 40
    BELT_Y_START = (SCREEN_HEIGHT - (BELT_COUNT * BELT_HEIGHT + (BELT_COUNT - 1) * BELT_SPACING)) // 2

    CUBE_SIZE = 30
    CUBE_SPACING = 120  # Distance between spawned cubes

    TARGET_X = SCREEN_WIDTH // 2
    TARGET_WIDTH = 4

    # Speeds are in pixels per second
    MAX_SPEED_PIXELS = 400.0
    MIN_SPEED_PIXELS = 0.0
    SPEED_ADJUSTMENT = 100.0

    SYNC_GOAL_SECONDS = 15
    SYNC_GOAL_FRAMES = int(SYNC_GOAL_SECONDS * FPS)
    MAX_EPISODE_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 64)

        # Initialize state variables to prevent errors before first reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.belt_speeds = np.zeros(self.BELT_COUNT, dtype=np.float32)
        self.belts = []
        self.sync_timer = 0
        self.was_fully_synced = False
        
        # Note: self.reset() is called by the wrapper, no need to call it here.
        # self.validate_implementation() # This is a self-test and not needed for final env.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sync_timer = 0
        self.was_fully_synced = False

        initial_speed = 60.0 # pixels per second
        self.belt_speeds = np.full(self.BELT_COUNT, initial_speed, dtype=np.float32)

        self.belts = []
        for i in range(self.BELT_COUNT):
            belt = {'cubes': []}
            # Pre-populate belts with cubes
            num_initial_cubes = int(self.SCREEN_WIDTH / self.CUBE_SPACING) + 2
            for j in range(num_initial_cubes):
                belt['cubes'].append({
                    'x': float(self.SCREEN_WIDTH - j * self.CUBE_SPACING),
                    'color_idx': self.np_random.integers(0, len(self.CUBE_COLORS))
                })
            self.belts.append(belt)
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_action(action)
        self._update_cubes()
        
        reward = self._calculate_reward_and_sync()
        self.score += reward
        
        terminated = self._check_termination()

        if terminated and self.sync_timer >= self.SYNC_GOAL_FRAMES:
            win_reward = 100.0
            self.score += win_reward
            reward += win_reward
        
        self.steps += 1
        
        # Truncation is not used in this game logic, but could be added for time limits.
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Belt 1 control (Up/Down)
        if movement == 1: self.belt_speeds[0] += self.SPEED_ADJUSTMENT * self.DT
        elif movement == 2: self.belt_speeds[0] -= self.SPEED_ADJUSTMENT * self.DT
        
        # Belt 2 control (Left/Right)
        if movement == 3: self.belt_speeds[1] += self.SPEED_ADJUSTMENT * self.DT
        elif movement == 4: self.belt_speeds[1] -= self.SPEED_ADJUSTMENT * self.DT

        # Belt 3 control (Space/Shift)
        if space_held: self.belt_speeds[2] += self.SPEED_ADJUSTMENT * self.DT
        if shift_held: self.belt_speeds[2] -= self.SPEED_ADJUSTMENT * self.DT

        # Belt 4 is uncontrolled and acts as a reference.
        
        # Clamp all controllable speeds
        np.clip(self.belt_speeds[:3], self.MIN_SPEED_PIXELS, self.MAX_SPEED_PIXELS, out=self.belt_speeds[:3])

    def _update_cubes(self):
        for i in range(self.BELT_COUNT):
            speed = self.belt_speeds[i]
            belt = self.belts[i]
            
            # Move existing cubes
            for cube in belt['cubes']:
                cube['x'] -= speed * self.DT # Cubes move from right to left
            
            # Remove off-screen cubes
            belt['cubes'] = [c for c in belt['cubes'] if c['x'] > -self.CUBE_SIZE]
            
            # Add new cubes if there's space on the right
            if not belt['cubes'] or belt['cubes'][-1]['x'] < self.SCREEN_WIDTH - self.CUBE_SPACING:
                # sfx: new_cube_spawn.wav
                belt['cubes'].append({
                    'x': float(self.SCREEN_WIDTH),
                    'color_idx': self.np_random.integers(0, len(self.CUBE_COLORS))
                })

    def _get_belt_colors_at_target(self):
        colors = [None] * self.BELT_COUNT
        for i in range(self.BELT_COUNT):
            for cube in self.belts[i]['cubes']:
                if cube['x'] <= self.TARGET_X < cube['x'] + self.CUBE_SIZE:
                    colors[i] = cube['color_idx']
                    break
        return colors

    def _calculate_reward_and_sync(self):
        colors_at_target = self._get_belt_colors_at_target()
        valid_colors = [c for c in colors_at_target if c is not None]
        
        reward = 0.0
        is_fully_synced = False

        if len(valid_colors) == self.BELT_COUNT:
            color_counts = Counter(valid_colors)
            most_common = color_counts.most_common(1)[0]
            
            if most_common[1] == 4:
                # sfx: sync_perfect.wav
                reward = 1.0
                self.sync_timer += 1
                is_fully_synced = True
            elif most_common[1] >= 2 or (len(color_counts) == 2 and list(color_counts.values()) == [2, 2]):
                # sfx: sync_partial.wav
                reward = 0.1
                self.sync_timer = 0
            else:
                self.sync_timer = 0
        else:
            self.sync_timer = 0
        
        # Difficulty increase on new full sync event
        if is_fully_synced and not self.was_fully_synced:
            # sfx: difficulty_up.wav
            self.belt_speeds *= 1.05
        
        self.was_fully_synced = is_fully_synced
        return reward

    def _check_termination(self):
        if self.sync_timer >= self.SYNC_GOAL_FRAMES:
            self.game_over = True
            return True
        # Max steps handled by truncation
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, x, y, font, color=COLOR_TEXT, shadow=True, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        
        if shadow:
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        
        self.screen.blit(text_surface, text_rect)

    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Belts and Cubes
        for i in range(self.BELT_COUNT):
            belt_y = self.BELT_Y_START + i * (self.BELT_HEIGHT + self.BELT_SPACING)
            pygame.draw.rect(self.screen, self.COLOR_BELT, (0, belt_y, self.SCREEN_WIDTH, self.BELT_HEIGHT))
            
            for cube in self.belts[i]['cubes']:
                cube_y = belt_y + (self.BELT_HEIGHT - self.CUBE_SIZE) // 2
                color = self.CUBE_COLORS[cube['color_idx']]
                
                shadow_rect = (int(cube['x'])-2, cube_y-2, self.CUBE_SIZE+4, self.CUBE_SIZE+4)
                # pygame.draw.rect(self.screen, (0,0,0,50), shadow_rect, border_radius=5)
                
                cube_rect = pygame.Rect(int(cube['x']), cube_y, self.CUBE_SIZE, self.CUBE_SIZE)
                pygame.draw.rect(self.screen, color, cube_rect, border_radius=4)
                
                highlight_color = tuple(min(255, c + 60) for c in color)
                pygame.draw.line(self.screen, highlight_color, (cube_rect.left+3, cube_rect.top+3), (cube_rect.right-4, cube_rect.top+3), 2)

        # Target Zone with glow effect
        glow_radius = self.TARGET_WIDTH * 4
        s = pygame.Surface((glow_radius * 2, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_TARGET_LINE, 30), s.get_rect(), border_radius=10)
        self.screen.blit(s, (self.TARGET_X - glow_radius, 0))

        pygame.draw.rect(self.screen, self.COLOR_TARGET_LINE, (self.TARGET_X - self.TARGET_WIDTH // 2, 0, self.TARGET_WIDTH, self.SCREEN_HEIGHT))

    def _render_ui(self):
        self._render_text(f"SCORE: {self.score:.1f}", 10, 10, self.font_medium)
        self._render_text(f"STEP: {self.steps}/{self.MAX_EPISODE_STEPS}", self.SCREEN_WIDTH - 200, 10, self.font_medium)

        for i in range(self.BELT_COUNT):
            belt_y = self.BELT_Y_START + i * (self.BELT_HEIGHT + self.BELT_SPACING)
            # Display speed in abstract units/sec for better readability
            speed_units = self.belt_speeds[i] / self.CUBE_SIZE 
            self._render_text(f"Belt {i+1} Speed: {speed_units:.1f}", 15, belt_y - 22, self.font_small)

        if self.sync_timer > 0:
            progress = self.sync_timer / self.SYNC_GOAL_FRAMES
            bar_width = int(progress * (self.SCREEN_WIDTH - 20))
            bar_y = self.SCREEN_HEIGHT - 30
            
            pygame.draw.rect(self.screen, self.COLOR_BELT, (10, bar_y, self.SCREEN_WIDTH - 20, 20), border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_TARGET_LINE, (10, bar_y, bar_width, 20), border_radius=5)
            self._render_text("SYNCHRONIZATION", self.SCREEN_WIDTH // 2, bar_y + 10, self.font_small, center=True)

        if self.game_over and self.sync_timer >= self.SYNC_GOAL_FRAMES:
             self._render_text("HARMONY ACHIEVED", self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, self.font_large, self.COLOR_TARGET_LINE, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sync_timer": self.sync_timer,
            "belt_speeds": self.belt_speeds.tolist()
        }

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

if __name__ == '__main__':
    # Example of how to run the environment with human controls
    # This part requires a display. If you run this, unset SDL_VIDEODRIVER
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Conveyor Harmony")
    clock = pygame.time.Clock()

    action = env.action_space.sample() 
    action.fill(0) # Start with a null action

    print("\n--- Human Controls ---")
    print("W/S:      Control Belt 1 Speed (Up/Down)")
    print("A/D:      Control Belt 2 Speed (Left/Right)")
    print("SPACE:    Increase Belt 3 Speed")
    print("L-SHIFT:  Decrease Belt 3 Speed")
    print("----------------------\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        action.fill(0) # Reset actions each frame
        
        # Belt 1
        if keys[pygame.K_w] or keys[pygame.K_UP]: action[0] = 1 # Up
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: action[0] = 2 # Down
        
        # Belt 2 - use the same movement action component
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: action[0] = 3 # Right (map to action 3)
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: action[0] = 4 # Left (map to action 4)
        
        # Belt 3
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0

        # The observation is already a rendered frame, just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()