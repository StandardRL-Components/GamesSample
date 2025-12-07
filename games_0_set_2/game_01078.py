
# Generated: 2025-08-27T15:48:25.979476
# Source Brief: brief_01078.md
# Brief Index: 1078

        
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
        "Controls: Use arrow keys to move your character on the grid. "
        "Collect all the gems before time runs out or you lose all your lives."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced grid-based puzzle game. Collect all the yellow gems while "
        "dodging the red enemy triangles. Can you get them all in time?"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 20
    GRID_ROWS = 12
    UI_HEIGHT = 40
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = (SCREEN_HEIGHT - UI_HEIGHT) // GRID_ROWS

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255, 100) # RGBA
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_GLOW = (255, 240, 100, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_UI_BG = (10, 15, 20)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEART = (255, 0, 80)

    # Game Parameters
    MAX_STEPS = 6000 # 60 seconds at 100 steps/sec
    START_LIVES = 3
    NUM_GEMS = 20
    NUM_ENEMIES = 5
    ENEMY_MOVE_INTERVAL = 5 # Moves every 5 steps

    # Reward Structure
    REWARD_MOVE_TOWARDS_GEM = 1.0
    REWARD_MOVE_AWAY_GEM = -0.1
    REWARD_MOVE_TOWARDS_ENEMY = -1.0
    REWARD_COLLECT_GEM = 10.0
    REWARD_LOSE_LIFE = -30.0
    REWARD_WIN = 50.0
    REWARD_LOSE = -100.0


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 52)


        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.gems_collected = 0
        self.player_pos = (0, 0)
        self.gem_positions = []
        self.enemies = []
        self.np_random = None
        self.win_message = ""

        self.reset()

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        col, row = grid_pos
        x = col * self.CELL_WIDTH + self.CELL_WIDTH / 2
        y = self.UI_HEIGHT + row * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        return int(x), int(y)

    def _get_manhattan_distance(self, pos1, pos2):
        """Calculates Manhattan distance between two grid positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_closest(self, start_pos, target_list):
        """Finds the closest target from a list to a starting position."""
        if not target_list:
            return None, float('inf')
        closest_target = None
        min_dist = float('inf')
        for target in target_list:
            dist = self._get_manhattan_distance(start_pos, target)
            if dist < min_dist:
                min_dist = dist
                closest_target = target
        return closest_target, min_dist

    def _generate_patrol_path(self, start_pos, max_size=4):
        """Generates a rectangular patrol path for an enemy."""
        path = [start_pos]
        x, y = start_pos
        size_x = self.np_random.integers(1, max_size + 1)
        size_y = self.np_random.integers(1, max_size + 1)

        # Move right
        for i in range(size_x):
            path.append((min(self.GRID_COLS - 1, x + i + 1), y))
        # Move down
        for i in range(size_y):
            path.append((path[-1][0], min(self.GRID_ROWS - 1, y + i + 1)))
        # Move left
        for i in range(size_x):
            path.append((max(0, path[-1][0] - 1), path[-1][1]))
        # Move up
        for i in range(size_y):
            path.append((path[-1][0], max(0, path[-1][1] - 1)))
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.START_LIVES
        self.gems_collected = 0

        # Player placement
        self.player_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)

        # Generate all possible grid positions
        all_positions = [(c, r) for c in range(self.GRID_COLS) for r in range(self.GRID_ROWS)]
        all_positions.remove(self.player_pos)
        self.np_random.shuffle(all_positions)

        # Gem placement
        self.gem_positions = all_positions[:self.NUM_GEMS]
        occupied_positions = set(self.gem_positions)
        occupied_positions.add(self.player_pos)

        # Enemy placement
        self.enemies = []
        available_positions = [p for p in all_positions if p not in occupied_positions]
        self.np_random.shuffle(available_positions)

        for i in range(min(self.NUM_ENEMIES, len(available_positions))):
            start_pos = available_positions[i]
            path = self._generate_patrol_path(start_pos)
            self.enemies.append({
                "pos": start_pos,
                "path": path,
                "path_index": 0
            })
            occupied_positions.add(start_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1 # Not used
        # shift_held = action[2] == 1 # Not used

        reward = 0
        
        # --- Pre-move calculations for reward ---
        old_player_pos = self.player_pos
        enemy_positions = [e['pos'] for e in self.enemies]
        _, old_dist_gem = self._find_closest(old_player_pos, self.gem_positions)
        _, old_dist_enemy = self._find_closest(old_player_pos, enemy_positions)

        # --- Player Movement ---
        px, py = self.player_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        self.player_pos = (
            max(0, min(self.GRID_COLS - 1, px)),
            max(0, min(self.GRID_ROWS - 1, py))
        )
        
        # --- Post-move calculations for reward ---
        _, new_dist_gem = self._find_closest(self.player_pos, self.gem_positions)
        _, new_dist_enemy = self._find_closest(self.player_pos, enemy_positions)
        
        # Movement-based rewards
        if new_dist_gem < old_dist_gem:
            reward += self.REWARD_MOVE_TOWARDS_GEM
        elif new_dist_gem > old_dist_gem:
            reward += self.REWARD_MOVE_AWAY_GEM

        if new_dist_enemy < old_dist_enemy:
            reward += self.REWARD_MOVE_TOWARDS_ENEMY

        # --- Enemy Movement ---
        if self.steps > 0 and self.steps % self.ENEMY_MOVE_INTERVAL == 0:
            for enemy in self.enemies:
                if enemy["path"]: # Check if path is not empty
                    enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
                    enemy['pos'] = enemy['path'][enemy['path_index']]
                    # Sound: Enemy move
        
        # --- Collision Detection ---
        # With Gems
        if self.player_pos in self.gem_positions:
            self.gem_positions.remove(self.player_pos)
            self.gems_collected += 1
            reward += self.REWARD_COLLECT_GEM
            self.score += self.REWARD_COLLECT_GEM
            # Sound: Gem collect

        # With Enemies
        for enemy in self.enemies:
            if self.player_pos == enemy['pos']:
                self.lives -= 1
                reward += self.REWARD_LOSE_LIFE
                self.score += self.REWARD_LOSE_LIFE
                # Reset player to center to avoid death loop
                self.player_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
                # Sound: Player hit
                break

        # --- Update State ---
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if self.gems_collected >= self.NUM_GEMS:
            terminated = True
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN
            self.game_over = True
            self.win_message = "YOU WIN!"
        elif self.lives <= 0:
            terminated = True
            reward += self.REWARD_LOSE
            self.score += self.REWARD_LOSE
            self.game_over = True
            self.win_message = "GAME OVER"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "TIME'S UP"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Draw grid
        for c in range(self.GRID_COLS + 1):
            x = c * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.UI_HEIGHT), (x, self.SCREEN_HEIGHT))
        for r in range(self.GRID_ROWS + 1):
            y = self.UI_HEIGHT + r * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw gems
        for pos in self.gem_positions:
            px, py = self._grid_to_pixel(pos)
            w, h = self.CELL_WIDTH * 0.3, self.CELL_HEIGHT * 0.4
            points = [(px, py - h), (px + w, py), (px, py + h), (px - w, py)]
            
            # Glow effect
            glow_surface = pygame.Surface((int(w*2.5), int(h*2.5)), pygame.SRCALPHA)
            radius = max(w,h) * (1 + 0.2 * math.sin(self.steps * 0.1))
            pygame.draw.circle(glow_surface, self.COLOR_GEM_GLOW, (w*1.25, h*1.25), radius)
            self.screen.blit(glow_surface, (px - w*1.25, py - h*1.25))
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)

        # Draw enemies
        for enemy in self.enemies:
            px, py = self._grid_to_pixel(enemy['pos'])
            w, h = self.CELL_WIDTH * 0.35, self.CELL_HEIGHT * 0.35
            points = [(px, py - h), (px - w, py + h), (px + w, py + h)]
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Draw player
        px, py = self._grid_to_pixel(self.player_pos)
        size = self.CELL_WIDTH * 0.7
        player_rect = pygame.Rect(px - size/2, py - size/2, size, size)
        
        # Glow effect
        glow_surface = pygame.Surface((int(size*2), int(size*2)), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (size, size), size * 0.8)
        self.screen.blit(glow_surface, (player_rect.x - size/2, player_rect.y - size/2))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)


    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT - 1), (self.SCREEN_WIDTH, self.UI_HEIGHT - 1))

        # Gem Count
        gem_text = self.font_main.render(f"GEMS: {self.gems_collected}/{self.NUM_GEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (10, 10))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / 100.0)
        time_text = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 10))

        # Lives
        lives_text = self.font_main.render("LIVES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.lives):
            self._draw_heart(self.SCREEN_WIDTH - 80 + i * 25, 20, 10)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _draw_heart(self, x, y, size):
        """Draws a heart shape for the lives indicator."""
        p1 = (x, y + size * 0.25)
        p2 = (x - size * 0.5, y - size * 0.25)
        p3 = (x - size * 0.25, y - size * 0.6)
        p4 = (x, y - size * 0.25)
        p5 = (x + size * 0.25, y - size * 0.6)
        p6 = (x + size * 0.5, y - size * 0.25)
        points = [p1, p2, p3, p4, p5, p6]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)


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
            "lives": self.lives,
            "gems_collected": self.gems_collected,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example of how to run the environment ---
if __name__ == '__main__':
    import os
    # Set this to 'human' to see the game being played.
    # Set it to 'rgb_array' for headless execution.
    # Use an environment variable to set the mode, default to 'rgb_array'
    RENDER_MODE = os.environ.get('RENDER_MODE', 'rgb_array')

    if RENDER_MODE == 'human':
        import sys
        
        env = GameEnv(render_mode='rgb_array')
        env.reset()
        
        # Override screen for display
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Gem Collector")
        clock = pygame.time.Clock()
        
        running = True
        last_action_time = -1
        action_interval = 100 # ms between actions
        
        while running:
            movement = 0 # No-op
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Allow continuous key presses but with a delay
            if current_time - last_action_time > action_interval:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                elif keys[pygame.K_ESCAPE]: running = False
            
            if movement != 0:
                last_action_time = current_time
                action = [movement, 0, 0] # Space/Shift not used
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}")
                    frame = env._get_observation()
                    frame = np.transpose(frame, (1, 0, 2))
                    surf = pygame.surfarray.make_surface(frame)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    pygame.time.wait(3000) # Wait 3 seconds
                    env.reset()

            # Always render the current state
            frame = env._get_observation()
            frame = np.transpose(frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit visual frame rate
            
        env.close()
        pygame.quit()
        sys.exit()

    else: # rgb_array mode for testing
        env = GameEnv(render_mode='rgb_array')
        env.validate_implementation()
        
        obs, info = env.reset()
        print("Initial state:", info)
        
        total_reward = 0
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if i % 20 == 0:
                print(f"Step {i}: Action={action}, Reward={reward:.2f}, Info={info}")
            if terminated:
                print(f"Episode finished after {i+1} steps. Final Info: {info}")
                break
        
        print(f"Total reward from random agent: {total_reward}")
        env.close()