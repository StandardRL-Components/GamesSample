
# Generated: 2025-08-27T16:57:07.226537
# Source Brief: brief_01380.md
# Brief Index: 1380

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Collect all the gems to win, but avoid the traps!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, collecting gems while avoiding deadly traps to achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.MAZE_WIDTH = self.WIDTH // self.GRID_SIZE
        self.MAZE_HEIGHT = self.HEIGHT // self.GRID_SIZE
        self.WIN_SCORE = 100
        self.MAX_STEPS = 1000
        self.INITIAL_TRAPS = 3
        self.INITIAL_GEMS = 5

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_game_over = pygame.font.SysFont("Consolas", 64, bold=True)

        # Colors
        self.COLOR_BG = (34, 40, 49) # #222831
        self.COLOR_WALL = (57, 62, 70) # #393E46
        self.COLOR_PLAYER = (255, 215, 0) # #FFD700
        self.COLOR_PLAYER_GLOW = (255, 215, 0, 60)
        self.COLOR_TRAP = (192, 57, 43) # #C0392B
        self.COLOR_GEM_PALETTE = [(231, 76, 60), (46, 204, 113), (52, 152, 219)] # Red, Green, Blue
        self.COLOR_TEXT = (238, 238, 238) # #EEEEEE
        
        # State variables are initialized in reset()
        self.maze = None
        self.player_pos = None
        self.gem_pos = None
        self.trap_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self._generate_maze()
        
        valid_spawns = self._get_valid_spawn_points()
        
        # Player spawn
        player_idx = self.np_random.choice(len(valid_spawns))
        self.player_pos = valid_spawns.pop(player_idx)
        
        # Trap and Gem spawn
        num_traps = self.INITIAL_TRAPS
        num_gems = self.INITIAL_GEMS
        
        # Ensure we don't try to spawn more items than available spaces
        num_to_spawn = min(len(valid_spawns), num_traps + num_gems)
        spawn_indices = self.np_random.choice(len(valid_spawns), size=num_to_spawn, replace=False)
        
        self.trap_pos = [valid_spawns[i] for i in spawn_indices[:num_traps]]
        self.gem_pos = []
        for i in spawn_indices[num_traps:num_to_spawn]:
            gem_type = self.np_random.integers(0, len(self.COLOR_GEM_PALETTE))
            self.gem_pos.append({"pos": valid_spawns[i], "type": gem_type})
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        reward = 0

        # --- Reward for moving closer/further from objectives ---
        dist_gem_before = self._find_closest_distance(self.player_pos, [g['pos'] for g in self.gem_pos])
        dist_trap_before = self._find_closest_distance(self.player_pos, self.trap_pos)

        # --- Player Movement ---
        px, py = self.player_pos
        new_px, new_py = px, py
        if movement == 1: # Up
            new_py -= 1
        elif movement == 2: # Down
            new_py += 1
        elif movement == 3: # Left
            new_px -= 1
        elif movement == 4: # Right
            new_px += 1
        
        if self.maze[new_py][new_px] == 0: # 0 is path
            self.player_pos = (new_px, new_py)

        # --- Calculate distance-based reward ---
        dist_gem_after = self._find_closest_distance(self.player_pos, [g['pos'] for g in self.gem_pos])
        dist_trap_after = self._find_closest_distance(self.player_pos, self.trap_pos)

        if dist_gem_before is not None and dist_gem_after is not None:
            # Reward for getting closer to a gem
            reward += (dist_gem_before - dist_gem_after)
        if dist_trap_before is not None and dist_trap_after is not None:
            # Penalty for getting closer to a trap
            reward -= 0.1 * (dist_trap_before - dist_trap_after)
        
        # --- Check for interactions ---
        # Gem collection
        collected_gem_idx = -1
        for i, gem in enumerate(self.gem_pos):
            if self.player_pos == gem['pos']:
                collected_gem_idx = i
                break
        
        if collected_gem_idx != -1:
            # Sound: gem_collect.wav
            self.score += 10
            reward += 10
            self.gem_pos.pop(collected_gem_idx)
            self._spawn_one_gem()

            # Increase trap count every 25 score
            current_trap_level = self.score // 25
            expected_traps = self.INITIAL_TRAPS + current_trap_level
            if len(self.trap_pos) < expected_traps:
                self._spawn_one_trap()

        # Trap collision
        if self.player_pos in self.trap_pos:
            # Sound: player_death.wav
            self.game_over = True
            reward = -100

        # --- Check for termination conditions ---
        terminated = self.game_over
        if self.score >= self.WIN_SCORE:
            # Sound: victory.wav
            self.win = True
            terminated = True
            if reward > -100: # Don't overwrite death penalty
                reward = 100
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render maze
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[y][x] == 1: # Wall
                    pygame.draw.rect(
                        self.screen, self.COLOR_WALL,
                        (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                    )
        
        # Render traps
        for tx, ty in self.trap_pos:
            center_x = int((tx + 0.5) * self.GRID_SIZE)
            center_y = int((ty + 0.5) * self.GRID_SIZE)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            size = int(self.GRID_SIZE * 0.3 + pulse * self.GRID_SIZE * 0.1)
            
            pygame.draw.line(self.screen, self.COLOR_TRAP, (center_x - size, center_y - size), (center_x + size, center_y + size), 3)
            pygame.draw.line(self.screen, self.COLOR_TRAP, (center_x - size, center_y + size), (center_x + size, center_y - size), 3)

        # Render gems
        for gem in self.gem_pos:
            gx, gy = gem['pos']
            color = self.COLOR_GEM_PALETTE[gem['type']]
            rect = pygame.Rect(gx * self.GRID_SIZE, gy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            
            # Pulsing glow effect
            pulse = (math.sin(self.steps * 0.15 + gx + gy) + 1) / 2
            glow_size = int(self.GRID_SIZE * (1.2 + pulse * 0.4))
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_size // 2, glow_size // 2, glow_size // 2, (*color, 30))
            self.screen.blit(glow_surf, (rect.centerx - glow_size // 2, rect.centery - glow_size // 2))

            pygame.draw.rect(self.screen, color, rect.inflate(-self.GRID_SIZE*0.4, -self.GRID_SIZE*0.4))

        # Render player
        px, py = self.player_pos
        player_rect = pygame.Rect(px * self.GRID_SIZE, py * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        # Player glow
        glow_size = int(self.GRID_SIZE * 2.5)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_size//2, glow_size//2, glow_size//2, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size//2, player_rect.centery - glow_size//2))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-self.GRID_SIZE*0.2, -self.GRID_SIZE*0.2))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_PLAYER)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_TRAP)
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _generate_maze(self):
        # Maze is represented as a grid: 1 for wall, 0 for path
        self.maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.int8)
        
        start_x = self.np_random.integers(0, self.MAZE_WIDTH // 2) * 2 + 1
        start_y = self.np_random.integers(0, self.MAZE_HEIGHT // 2) * 2 + 1
        
        stack = deque([(start_x, start_y)])
        self.maze[start_y, start_x] = 0

        while stack:
            x, y = stack[-1]
            
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.MAZE_WIDTH - 1 and 0 < ny < self.MAZE_HEIGHT - 1 and self.maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                self.maze[ny, nx] = 0
                self.maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _get_valid_spawn_points(self):
        points = []
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[y][x] == 0:
                    points.append((x, y))
        return points

    def _spawn_one_gem(self):
        valid_spawns = self._get_valid_spawn_points()
        occupied = [g['pos'] for g in self.gem_pos] + self.trap_pos + [self.player_pos]
        available_spawns = [p for p in valid_spawns if p not in occupied]
        if not available_spawns: return
        spawn_pos = available_spawns[self.np_random.integers(len(available_spawns))]
        gem_type = self.np_random.integers(len(self.COLOR_GEM_PALETTE))
        self.gem_pos.append({"pos": spawn_pos, "type": gem_type})

    def _spawn_one_trap(self):
        valid_spawns = self._get_valid_spawn_points()
        occupied = [g['pos'] for g in self.gem_pos] + self.trap_pos + [self.player_pos]
        available_spawns = [p for p in valid_spawns if p not in occupied]
        if not available_spawns: return
        spawn_pos = available_spawns[self.np_random.integers(len(available_spawns))]
        self.trap_pos.append(spawn_pos)

    def _find_closest_distance(self, pos, targets):
        if not targets:
            return None
        px, py = pos
        min_dist = float('inf')
        for tx, ty in targets:
            dist = abs(px - tx) + abs(py - ty) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Maze Gem Collector")
    
    terminated = False
    
    # Game loop
    while not terminated:
        action = np.array([0, 0, 0]) # Default action: no-op
        move_made = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                # Set move_made to False to exit loop cleanly
                move_made = False
                continue

            if event.type == pygame.KEYDOWN:
                move_made = True
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    move_made = False # Don't step on reset
                else:
                    move_made = False # Don't step on other key presses

        if move_made:
            obs, reward, term, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")
            if term:
                # Render one last time to show final state
                frame_to_show = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame_to_show)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                print("Game Over!")
                pygame.time.wait(2000) # Pause for 2 seconds
                
                # Reset for a new game
                obs, info = env.reset()

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it.
        frame_to_show = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_show)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
            
    env.close()