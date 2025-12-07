
# Generated: 2025-08-28T06:19:10.460020
# Source Brief: brief_05858.md
# Brief Index: 5858

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your light source one square at a time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a haunted forest grid, evading lurking creatures to reach the safety of the clearing. Your light reveals only the immediate surroundings."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 50
        self.WORLD_HEIGHT = 40
        self.CELL_SIZE = 40
        self.VIEW_RADIUS = 4 # 9x9 grid is a radius of 4
        self.MAX_STEPS = 1000
        self.N_CREATURES = 7
        self.TREE_DENSITY = 0.25
        self.MIN_CLEARING_DISTANCE = 25

        # Colors
        self.COLOR_BG = (10, 15, 10) # Very dark green for fog
        self.COLOR_GROUND = (25, 35, 25) # Dark green for visible ground
        self.COLOR_TREE = (15, 50, 15)
        self.COLOR_PLAYER = (255, 255, 220)
        self.COLOR_PLAYER_GLOW = (200, 200, 150)
        self.COLOR_CREATURE = (255, 50, 50)
        self.COLOR_CREATURE_FLICKER = (180, 20, 20)
        self.COLOR_CLEARING = (180, 255, 180)
        self.COLOR_CLEARING_GLOW = (100, 200, 100)
        self.COLOR_TEXT = (220, 220, 220)

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
        self.font = pygame.font.Font(None, 28)
        
        # Initialize state variables to be populated in reset()
        self.player_pos = (0, 0)
        self.clearing_pos = (0, 0)
        self.creatures = []
        self.grid = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH), dtype=np.int8)
        self.visibility_grid = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH), dtype=bool)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.distance_to_clearing = 0
        
        # Initialize state for validation
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_world()
        
        self.distance_to_clearing = self._manhattan_distance(self.player_pos, self.clearing_pos)
        self._update_visibility()
        
        return self._get_observation(), self._get_info()

    def _generate_world(self):
        self.grid = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH), dtype=np.int8)
        self.visibility_grid = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH), dtype=bool)

        # Place trees
        for y in range(self.WORLD_HEIGHT):
            for x in range(self.WORLD_WIDTH):
                if self.np_random.random() < self.TREE_DENSITY:
                    self.grid[y, x] = 1 # 1 is a tree

        # Place player
        self.player_pos = self._get_random_empty_cell()

        # Place clearing
        while True:
            self.clearing_pos = self._get_random_empty_cell()
            if self._manhattan_distance(self.player_pos, self.clearing_pos) >= self.MIN_CLEARING_DISTANCE:
                break
        
        # Place creatures
        self.creatures = []
        for _ in range(self.N_CREATURES):
            while True:
                path = self._generate_patrol_path()
                if path:
                    # Ensure creature starts far from player
                    if self._manhattan_distance(path[0], self.player_pos) > 10:
                        self.creatures.append({"path": path, "path_index": 0, "pos": path[0]})
                        break

    def _get_random_empty_cell(self):
        while True:
            x = self.np_random.integers(0, self.WORLD_WIDTH)
            y = self.np_random.integers(0, self.WORLD_HEIGHT)
            if self.grid[y, x] == 0:
                return (x, y)

    def _generate_patrol_path(self):
        start_pos = self._get_random_empty_cell()
        path = [start_pos]
        current_pos = start_pos
        
        path_len = self.np_random.integers(3, 6)
        for _ in range(path_len - 1):
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current_pos[0] + dx, current_pos[1] + dy
                if 0 <= nx < self.WORLD_WIDTH and 0 <= ny < self.WORLD_HEIGHT and self.grid[ny, nx] == 0:
                    if (nx, ny) not in path:
                        neighbors.append((nx, ny))
            
            if not neighbors:
                return None # Failed to generate path
            
            # np_random.choice on a list of tuples needs a workaround
            chosen_idx = self.np_random.integers(len(neighbors))
            current_pos = neighbors[chosen_idx]
            path.append(current_pos)
            
        # Create a back-and-forth patrol from the generated path
        if len(path) > 1:
            patrol_loop = path + path[-2:0:-1]
        else:
            patrol_loop = path
        return patrol_loop

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _update_visibility(self):
        px, py = self.player_pos
        for y in range(max(0, py - self.VIEW_RADIUS), min(self.WORLD_HEIGHT, py + self.VIEW_RADIUS + 1)):
            for x in range(max(0, px - self.VIEW_RADIUS), min(self.WORLD_WIDTH, px + self.VIEW_RADIUS + 1)):
                self.visibility_grid[y, x] = True

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # --- Player Movement ---
        old_dist = self._manhattan_distance(self.player_pos, self.clearing_pos)
        px, py = self.player_pos
        
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        # Check boundaries and obstacles
        if 0 <= px < self.WORLD_WIDTH and 0 <= py < self.WORLD_HEIGHT and self.grid[py, px] == 0:
            self.player_pos = (px, py)
            # sfx: player_step
        
        self._update_visibility()

        # --- Creature Movement ---
        for creature in self.creatures:
            if creature["path"]: # Ensure path is not empty
                creature["path_index"] = (creature["path_index"] + 1) % len(creature["path"])
                creature["pos"] = creature["path"][creature["path_index"]]
                # sfx: creature_hiss

        # --- State Update and Reward Calculation ---
        self.steps += 1
        reward = 0.0
        terminated = False

        # Check for win/loss conditions
        if self.player_pos == self.clearing_pos:
            reward = 100.0
            terminated = True
            self.game_over = True
            # sfx: win_jingle
        else:
            for creature in self.creatures:
                if self.player_pos == creature["pos"]:
                    reward = -100.0
                    terminated = True
                    self.game_over = True
                    # sfx: player_death
                    break
        
        if not terminated:
            new_dist = self._manhattan_distance(self.player_pos, self.clearing_pos)
            if new_dist < old_dist:
                reward = 0.1
            elif new_dist > old_dist:
                reward = -0.2
            
            self.distance_to_clearing = new_dist
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # The camera is centered on the player.
        cam_world_x = self.player_pos[0] - (self.SCREEN_WIDTH / 2 / self.CELL_SIZE)
        cam_world_y = self.player_pos[1] - (self.SCREEN_HEIGHT / 2 / self.CELL_SIZE)

        start_x = max(0, int(cam_world_x))
        end_x = min(self.WORLD_WIDTH, int(cam_world_x + self.SCREEN_WIDTH / self.CELL_SIZE) + 2)
        start_y = max(0, int(cam_world_y))
        end_y = min(self.WORLD_HEIGHT, int(cam_world_y + self.SCREEN_HEIGHT / self.CELL_SIZE) + 2)

        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if self.visibility_grid[y, x]:
                    screen_x = int((x - cam_world_x) * self.CELL_SIZE)
                    screen_y = int((y - cam_world_y) * self.CELL_SIZE)
                    
                    rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
                    
                    if (x, y) == self.clearing_pos:
                        glow_rect = rect.inflate(self.CELL_SIZE, self.CELL_SIZE)
                        pygame.draw.rect(self.screen, self.COLOR_CLEARING_GLOW, glow_rect, border_radius=int(self.CELL_SIZE/2))
                        pygame.draw.rect(self.screen, self.COLOR_CLEARING, rect, border_radius=int(self.CELL_SIZE/4))
                    elif self.grid[y, x] == 1: # Tree
                        pygame.draw.rect(self.screen, self.COLOR_TREE, rect)
                    else: # Empty ground
                        pygame.draw.rect(self.screen, self.COLOR_GROUND, rect)

        # Render creatures
        for creature in self.creatures:
            cx, cy = creature["pos"]
            if self.visibility_grid[cy, cx]:
                screen_x = int((cx - cam_world_x) * self.CELL_SIZE)
                screen_y = int((cy - cam_world_y) * self.CELL_SIZE)
                
                creature_rect = pygame.Rect(
                    screen_x + self.CELL_SIZE // 4, 
                    screen_y + self.CELL_SIZE // 4,
                    self.CELL_SIZE // 2,
                    self.CELL_SIZE // 2
                )
                
                color = self.COLOR_CREATURE if (self.steps // 3) % 2 == 0 else self.COLOR_CREATURE_FLICKER
                pygame.draw.rect(self.screen, color, creature_rect)

        # Render player (always in the center)
        player_screen_x = self.SCREEN_WIDTH // 2
        player_screen_y = self.SCREEN_HEIGHT // 2
        
        pygame.gfxdraw.filled_circle(self.screen, player_screen_x, player_screen_y, int(self.CELL_SIZE * 0.6), self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_screen_x, player_screen_y, int(self.CELL_SIZE * 0.35), self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_screen_x, player_screen_y, int(self.CELL_SIZE * 0.35), self.COLOR_PLAYER)

    def _render_ui(self):
        dist_text = f"Distance to Clearing: {self.distance_to_clearing}"
        text_surface = self.font.render(dist_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        score_text = f"Score: {self.score:.1f}"
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surface, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "distance_to_clearing": self.distance_to_clearing,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Haunted Forest")

    running = True
    terminated = False
    
    # Initial render
    obs, _ = env.reset()
    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    render_screen.blit(surf, (0, 0))
    pygame.display.flip()

    while running:
        action = 0 # No-op by default
        
        event_occurred = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                event_occurred = True
                if event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_RIGHT:
                    action = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    terminated = False
                    obs, _ = env.reset()
                    action = 0
                elif event.key == pygame.K_q:
                    running = False

        if event_occurred and not terminated:
            full_action = [action, 0, 0] 
            obs, reward, terminated, truncated, info = env.step(full_action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {info['score']:.1f}, Steps: {info['steps']}")
                print("Press 'r' to restart or 'q' to quit.")
        
        env.clock.tick(30)

    env.close()