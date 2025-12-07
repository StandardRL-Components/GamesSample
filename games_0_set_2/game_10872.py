import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:04:25.367613
# Source Brief: brief_00872.md
# Brief Index: 872
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a maze and collect items. Switch between small and large forms to grab different types of collectibles."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to switch between small and large size."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CRITICAL: Game Parameters ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 31, 19 # Odd numbers work best for generator
        self.TILE_SIZE = 20
        self.MAX_STEPS = 1000
        self.TARGET_SMALL_ITEMS = 10
        self.TARGET_LARGE_ITEMS = 5

        # --- CRITICAL: Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_size = pygame.font.SysFont("monospace", 14, bold=True)

        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_WALL = (40, 50, 90)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 40)
        self.COLOR_ITEM_SMALL = (255, 220, 0)
        self.COLOR_ITEM_LARGE = (255, 50, 50)
        self.COLOR_SPAWN_POINT = (50, 60, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_PARTICLE = (255, 255, 255)

        # --- State Variables ---
        self.maze = None
        self.player_pos = None
        self.render_pos = None
        self.player_is_large = None
        self.small_items = None
        self.large_items = None
        self.collected_small_count = None
        self.collected_large_count = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.last_space_held = None
        
        self.maze_offset_x = (self.WIDTH - self.MAZE_WIDTH * self.TILE_SIZE) // 2
        self.maze_offset_y = (self.HEIGHT - self.MAZE_HEIGHT * self.TILE_SIZE) // 2
        
        self.np_random = None # Will be seeded in reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        
        self.player_is_large = False
        self.collected_small_count = 0
        self.collected_large_count = 0

        self.particles = []

        self._generate_maze()
        self._spawn_player_and_items()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, True, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- State before action ---
        old_pos = self.player_pos[:]
        old_dist_to_target = self._get_distance_to_target(old_pos)

        # --- Action: Size Toggle ---
        # Toggle on press (rising edge)
        if space_held and not self.last_space_held:
            self.player_is_large = not self.player_is_large
            # Sound placeholder: pygame.mixer.Sound("size_change.wav").play()
        self.last_space_held = space_held

        # --- Action: Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            if self._is_valid_move(new_pos):
                self.player_pos = new_pos

        # --- Item Collection Logic ---
        collected_small_item = False
        collected_large_item = False

        if self.player_pos in self.small_items:
            self.small_items.remove(self.player_pos)
            self.collected_small_count += 1
            self.score += 1
            reward += 1.0
            collected_small_item = True
            # Sound placeholder: pygame.mixer.Sound("collect_small.wav").play()

        if self.player_is_large and self.player_pos in self.large_items:
            self.large_items.remove(self.player_pos)
            self.collected_large_count += 1
            self.score += 2
            reward += 2.0
            collected_large_item = True
            # Sound placeholder: pygame.mixer.Sound("collect_large.wav").play()
        
        if collected_small_item or collected_large_item:
            self._spawn_particles(self.player_pos)

        # --- Reward Shaping ---
        new_dist_to_target = self._get_distance_to_target(self.player_pos)
        if new_dist_to_target < old_dist_to_target:
            reward += 0.1
        elif new_dist_to_target > old_dist_to_target:
            reward -= 0.1

        # --- Update Game State ---
        self.steps += 1
        
        terminated = (self.collected_small_count >= self.TARGET_SMALL_ITEMS and 
                      self.collected_large_count >= self.TARGET_LARGE_ITEMS)
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over: # Win condition
            reward += 50.0
            self.score += 50.0
        
        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

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
            "small_items_collected": self.collected_small_count,
            "large_items_collected": self.collected_large_count,
            "player_pos": self.player_pos,
        }

    def _render_game(self):
        # --- Update render position for smooth animation ---
        target_pixel_pos = self._grid_to_pixel(self.player_pos)
        lerp_factor = 0.4 # Controls animation speed
        self.render_pos[0] = self.render_pos[0] * (1 - lerp_factor) + target_pixel_pos[0] * lerp_factor
        self.render_pos[1] = self.render_pos[1] * (1 - lerp_factor) + target_pixel_pos[1] * lerp_factor

        # --- Draw Maze and Items ---
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                px, py = self._grid_to_pixel((x, y))
                rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
                
                if self.maze[y, x] == 1: # Wall
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else: # Path
                    pygame.draw.circle(self.screen, self.COLOR_SPAWN_POINT, rect.center, 2)

                if [x, y] in self.small_items:
                    item_rect = pygame.Rect(0, 0, self.TILE_SIZE * 0.5, self.TILE_SIZE * 0.5)
                    item_rect.center = rect.center
                    pygame.draw.rect(self.screen, self.COLOR_ITEM_SMALL, item_rect)
                
                if [x, y] in self.large_items:
                    item_rect = pygame.Rect(0, 0, self.TILE_SIZE * 0.8, self.TILE_SIZE * 0.8)
                    item_rect.center = rect.center
                    pygame.draw.rect(self.screen, self.COLOR_ITEM_LARGE, item_rect)
        
        # --- Draw Particles ---
        self._update_and_draw_particles()

        # --- Draw Player ---
        player_size_multiplier = 0.8 if self.player_is_large else 0.5
        player_size = int(self.TILE_SIZE * player_size_multiplier)
        player_rect = pygame.Rect(0, 0, player_size, player_size)
        player_rect.center = (int(self.render_pos[0] + self.TILE_SIZE / 2), int(self.render_pos[1] + self.TILE_SIZE / 2))
        
        # Glow effect
        glow_radius = int(player_size * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, player_rect.centerx, player_rect.centery, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, player_rect.centerx, player_rect.centery, glow_radius, self.COLOR_PLAYER_GLOW)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        size_text_str = "LARGE" if self.player_is_large else "SMALL"
        size_color = self.COLOR_ITEM_LARGE if self.player_is_large else self.COLOR_ITEM_SMALL
        size_text = self.font_size.render(f"SIZE: {size_text_str}", True, size_color)
        self.screen.blit(size_text, (15, 35))

        items_text = self.font_ui.render(f"ITEMS: {self.collected_small_count}/{self.TARGET_SMALL_ITEMS} S | {self.collected_large_count}/{self.TARGET_LARGE_ITEMS} L", True, self.COLOR_UI_TEXT)
        text_rect = items_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(items_text, text_rect)

    def _is_valid_move(self, pos):
        x, y = pos
        if not (0 <= x < self.MAZE_WIDTH and 0 <= y < self.MAZE_HEIGHT):
            return False # Out of bounds
        if self.maze[y, x] == 1:
            return False # Wall collision
        if not self.player_is_large and [x, y] in self.large_items:
            return False # Small player blocked by large item
        return True

    def _generate_maze(self):
        self.maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        stack = [(0, 0)]
        self.maze[0, 0] = 0
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                rand_idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[rand_idx]
                # Carve path
                self.maze[ny, nx] = 0
                self.maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _spawn_player_and_items(self):
        path_tiles = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(path_tiles)
        
        # Player spawn (y, x format from argwhere)
        player_y, player_x = path_tiles.pop()
        self.player_pos = [player_x, player_y]
        self.render_pos = self._grid_to_pixel(self.player_pos)
        
        # Item spawns
        self.small_items = []
        for _ in range(self.TARGET_SMALL_ITEMS):
            if not path_tiles: break
            y, x = path_tiles.pop()
            self.small_items.append([x, y])

        self.large_items = []
        for _ in range(self.TARGET_LARGE_ITEMS):
            if not path_tiles: break
            y, x = path_tiles.pop()
            self.large_items.append([x, y])

    def _find_nearest_item(self, from_pos, item_list):
        if not item_list:
            return None, float('inf')
        
        min_dist = float('inf')
        nearest_item = None
        for item_pos in item_list:
            dist = abs(from_pos[0] - item_pos[0]) + abs(from_pos[1] - item_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_item = item_pos
        return nearest_item, min_dist

    def _get_distance_to_target(self, pos):
        # Determine appropriate target based on size and what's left to collect
        target_list = None
        if self.player_is_large and self.collected_large_count < self.TARGET_LARGE_ITEMS:
            target_list = self.large_items
        elif self.collected_small_count < self.TARGET_SMALL_ITEMS:
            target_list = self.small_items
        else: # If small items are done, large player can go for large items
            target_list = self.large_items

        if not target_list: # If no targets of primary type, check other type
            target_list = self.small_items if target_list == self.large_items else self.large_items

        if not target_list:
            return 0 # No items left

        _, dist = self._find_nearest_item(pos, target_list)
        return dist

    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        return [x * self.TILE_SIZE + self.maze_offset_x, y * self.TILE_SIZE + self.maze_offset_y]
    
    def _spawn_particles(self, grid_pos):
        pixel_pos = self._grid_to_pixel(grid_pos)
        center_x = pixel_pos[0] + self.TILE_SIZE / 2
        center_y = pixel_pos[1] + self.TILE_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': [center_x, center_y], 'vel': velocity, 'life': lifespan})

    def _update_and_draw_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 20))
                color = self.COLOR_PARTICLE + (alpha,)
                radius = int(p['life'] / 5)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), max(1, radius), color)

    def validate_implementation(self):
        # This method is for developer convenience and not part of the Gym API
        print("Running implementation validation...")
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), "Obs shape mismatch"
        assert obs.dtype == np.uint8, "Obs dtype mismatch"
        assert isinstance(info, dict), "Info should be a dict"
        
        # Test action space
        assert self.action_space.shape == (3,), "Action space shape mismatch"
        assert self.action_space.nvec.tolist() == [5, 2, 2], "Action space nvec mismatch"
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), "Step obs shape mismatch"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(term, bool), "Terminated should be a bool"
        assert isinstance(trunc, bool), "Truncated should be a bool"
        assert isinstance(info, dict), "Step info should be a dict"
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # Set a non-dummy driver for interactive mode
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    env.validate_implementation()
    obs, info = env.reset(seed=42)
    
    # --- Interactive Play ---
    pygame.display.set_caption("Maze Shifter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + GameEnv.game_description)
    print(GameEnv.user_guide)
    print("Objective: Collect 10 small (yellow) and 5 large (red) items.")
    print("Note: Large items can only be collected when you are large.")

    while running:
        # Default action is "do nothing"
        action = [0, 0, 0] # move, space, shift
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            end_reason = "won" if terminated else "timed out"
            print(f"Episode finished ({end_reason})! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset(seed=random.randint(0, 10000))
            total_reward = 0
            
        clock.tick(30) # Limit to 30 FPS for smooth interactive play

    pygame.quit()
    env.close()