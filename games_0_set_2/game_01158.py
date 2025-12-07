
# Generated: 2025-08-27T16:12:59.660903
# Source Brief: brief_01158.md
# Brief Index: 1158

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move your ship. Collect all the gems before time runs out, but watch out for the red asteroids!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade game. Navigate a field of asteroids to collect valuable gems. Your score is based on gems collected and your efficiency."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60

    COLOR_BG = (15, 15, 25) # Dark space blue
    COLOR_PLAYER = (255, 255, 0) # Bright Yellow
    COLOR_OBSTACLE = (255, 50, 50) # Bright Red
    COLOR_UI_TEXT = (240, 240, 240) # Off-white
    COLOR_BOUNDARY = (100, 100, 120) # Muted grey
    
    GEM_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 0),    # Lime
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]

    PLAYER_SPEED = 6
    PLAYER_RADIUS = 12
    GEM_SIZE = 10
    OBSTACLE_RADIUS = 15
    
    NUM_GEMS = 15
    NUM_OBSTACLES = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.player_pos = None
        self.gems = None
        self.obstacles = None
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.dist_to_nearest_gem = float('inf')
        self.victory = False

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_left = self.FPS * self.GAME_DURATION_SECONDS
        self.game_over = False
        self.victory = False

        # Place player in the center
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])

        # Generate non-overlapping entities
        spawn_area = pygame.Rect(30, 30, self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT - 60)
        occupied_spaces = [pygame.Rect(self.player_pos[0] - 20, self.player_pos[1] - 20, 40, 40)]

        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            while True:
                pos = (self.np_random.integers(spawn_area.left, spawn_area.right), 
                       self.np_random.integers(spawn_area.top, spawn_area.bottom))
                new_rect = pygame.Rect(pos[0] - self.OBSTACLE_RADIUS, pos[1] - self.OBSTACLE_RADIUS, 
                                       self.OBSTACLE_RADIUS * 2, self.OBSTACLE_RADIUS * 2)
                if not any(new_rect.colliderect(r) for r in occupied_spaces):
                    self.obstacles.append(np.array(pos, dtype=float))
                    occupied_spaces.append(new_rect)
                    break
        
        self.gems = []
        for i in range(self.NUM_GEMS):
            while True:
                pos = (self.np_random.integers(spawn_area.left, spawn_area.right), 
                       self.np_random.integers(spawn_area.top, spawn_area.bottom))
                new_rect = pygame.Rect(pos[0] - self.GEM_SIZE, pos[1] - self.GEM_SIZE, 
                                       self.GEM_SIZE * 2, self.GEM_SIZE * 2)
                if not any(new_rect.colliderect(r) for r in occupied_spaces):
                    color = self.GEM_COLORS[i % len(self.GEM_COLORS)]
                    self.gems.append({'pos': np.array(pos, dtype=float), 'color': color})
                    occupied_spaces.append(new_rect)
                    break

        self._update_dist_to_nearest_gem()

        return self._get_observation(), self._get_info()

    def _update_dist_to_nearest_gem(self):
        if not self.gems:
            self.dist_to_nearest_gem = 0
            return
        
        distances = [np.linalg.norm(self.player_pos - gem['pos']) for gem in self.gems]
        self.dist_to_nearest_gem = min(distances) if distances else 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        reward = 0
        dist_before = self.dist_to_nearest_gem

        # 1. Update Player Position
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Clamp player to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # 2. Distance-based reward
        self._update_dist_to_nearest_gem()
        dist_after = self.dist_to_nearest_gem
        
        if dist_after < dist_before:
            reward += 0.1 # Closer to a gem
        else:
            reward -= 0.2 # Further from a gem

        # 3. Check Collisions
        # Gems
        collected_gems = []
        for gem in self.gems:
            if np.linalg.norm(self.player_pos - gem['pos']) < self.PLAYER_RADIUS + self.GEM_SIZE / 2:
                collected_gems.append(gem)
                self.score += 10
                reward += 10
                # sfx: gem collect sound
        self.gems = [gem for gem in self.gems if gem not in collected_gems]

        # Obstacles
        for obs_pos in self.obstacles:
            if np.linalg.norm(self.player_pos - obs_pos) < self.PLAYER_RADIUS + self.OBSTACLE_RADIUS:
                self.game_over = True
                self.victory = False
                reward = -100 # Terminal penalty
                # sfx: explosion sound
                break
        
        # 4. Update Time and Steps
        self.steps += 1
        self.time_left -= 1
        
        # 5. Check Termination Conditions
        terminated = self.game_over
        if not terminated:
            if self.time_left <= 0:
                terminated = True
                self.game_over = True
                self.victory = False
                reward = -100 # Terminal penalty for timeout
                # sfx: timeout buzzer
            elif not self.gems:
                terminated = True
                self.game_over = True
                self.victory = True
                self.score += 50 # Victory bonus
                reward = 100 # Terminal reward for winning
                # sfx: victory fanfare

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _render_game(self):
        # Draw obstacles (asteroids)
        for pos in self.obstacles:
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE)

        # Draw gems
        for gem in self.gems:
            pos = gem['pos']
            color = gem['color']
            size = self.GEM_SIZE
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Draw player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        # Add a small "cockpit" for directionality illusion
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, 4, self.COLOR_BG)


    def _render_ui(self):
        # Draw boundaries
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)

        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_str = f"Time: {max(0, self.time_left // self.FPS):02d}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Gems remaining
        gems_str = f"Gems Left: {len(self.gems)}"
        gems_text = self.font_ui.render(gems_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(gems_text, (10, self.SCREEN_HEIGHT - gems_text.get_height() - 10))

        # Game Over / Victory Message
        if self.game_over:
            if self.victory:
                msg = "VICTORY!"
                color = (50, 255, 50) # Green
            else:
                msg = "GAME OVER"
                color = self.COLOR_OBSTACLE # Red
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (H, W, C)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "gems_left": len(self.gems),
            "victory": self.victory,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False # Start a new game
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep the window open for a few seconds to see the final screen
    if info['game_over']:
        pygame.time.wait(3000)

    env.close()