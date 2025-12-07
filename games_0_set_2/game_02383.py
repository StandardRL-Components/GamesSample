
# Generated: 2025-08-28T04:38:58.917603
# Source Brief: brief_02383.md
# Brief Index: 2383

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric horror maze environment where the player must evade enemies
    and reach the exit. The player has a limited field of view, adding to the
    tension and requiring careful, strategic movement.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate the maze. "
        "Evade the red enemies and reach the glowing white exit."
    )

    # Short, user-facing description of the game
    game_description = (
        "A tense, isometric horror game. Navigate a dark, procedurally generated "
        "maze, using shadows to your advantage. Evade patrolling enemies and find "
        "the exit to survive."
    )

    # The game state is static until an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game Constants ---
        self.MAZE_WIDTH = 25
        self.MAZE_HEIGHT = 25
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.TILE_WIDTH_HALF = self.TILE_WIDTH // 2
        self.TILE_HEIGHT_HALF = self.TILE_HEIGHT // 2
        self.NUM_ENEMIES = 5
        self.NUM_CHECKPOINTS = 3
        self.MAX_STEPS = 1000
        self.VISIBILITY_RADIUS = 5.5 * self.TILE_WIDTH

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_WALL = (45, 45, 60)
        self.COLOR_FLOOR = (70, 70, 90)
        self.COLOR_SHADOW = (55, 55, 70)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_CHECKPOINT = (255, 220, 0)
        self.COLOR_CHECKPOINT_REACHED = (100, 180, 100)
        self.COLOR_EXIT = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)

        # --- Tile Types ---
        self.TILE_EMPTY = 0
        self.TILE_WALL = 1
        self.TILE_SHADOW = 2
        
        # --- State Variables (initialized in reset) ---
        self.maze = None
        self.player_pos = None
        self.enemies = []
        self.checkpoints = []
        self.reached_checkpoints = []
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = "" # "VICTORY" or "DEFEAT"
        self.enemy_speed = 0.0
        self.camera_offset = [0, 0]
        self.rng = None

        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = (x - y) * self.TILE_WIDTH_HALF + self.camera_offset[0]
        screen_y = (x + y) * self.TILE_HEIGHT_HALF + self.camera_offset[1]
        return int(screen_x), int(screen_y)

    def _generate_maze(self):
        """Generates a maze using recursive backtracking and adds features."""
        self.maze = np.ones((self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.uint8) * self.TILE_WALL
        path_tiles = []

        def carve(x, y):
            path_tiles.append((x, y))
            self.maze[x, y] = self.TILE_EMPTY
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.rng.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[nx, ny] == self.TILE_WALL:
                    self.maze[x + dx, y + dy] = self.TILE_EMPTY
                    carve(nx, ny)

        start_x, start_y = self.rng.integers(1, self.MAZE_WIDTH, 2)
        start_x |= 1; start_y |= 1 # Ensure odd coordinates
        carve(start_x, start_y)

        # Place player at start of path generation
        self.player_pos = list(path_tiles[0])
        
        # Place exit at the end of path generation
        self.exit_pos = list(path_tiles[-1])
        
        # Designate shadow tiles
        num_shadows = int(len(path_tiles) * 0.3) # 30% of path tiles are shadows
        shadow_indices = self.rng.choice(len(path_tiles), num_shadows, replace=False)
        for i in shadow_indices:
            sx, sy = path_tiles[i]
            if (sx, sy) != self.player_pos and (sx, sy) != self.exit_pos:
                self.maze[sx, sy] = self.TILE_SHADOW
        
        return path_tiles

    def _find_path(self, start, end):
        """Finds a path using Breadth-First Search."""
        q = collections.deque([(start, [start])])
        visited = {start}
        while q:
            (vx, vy), path = q.popleft()
            if (vx, vy) == end:
                return path
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = vx + dx, vy + dy
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[nx, ny] != self.TILE_WALL and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
        return [start] # Should not happen in a connected maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        # --- Generate World ---
        path_tiles = self._generate_maze()
        
        # Place checkpoints
        self.checkpoints = []
        self.reached_checkpoints = []
        if len(path_tiles) > self.NUM_CHECKPOINTS + 2:
            checkpoint_indices = np.linspace(0, len(path_tiles) - 1, self.NUM_CHECKPOINTS + 2, dtype=int)[1:-1]
            for i in checkpoint_indices:
                self.checkpoints.append(list(path_tiles[i]))

        # Place enemies
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            while True:
                start_node = tuple(path_tiles[self.rng.integers(0, len(path_tiles))])
                end_node = tuple(path_tiles[self.rng.integers(0, len(path_tiles))])
                if start_node != self.player_pos and end_node != self.player_pos and np.linalg.norm(np.array(start_node) - np.array(end_node)) > 5:
                    break
            
            patrol_path = self._find_path(start_node, end_node)
            self.enemies.append({
                "pos": list(patrol_path[0]),
                "path": patrol_path,
                "path_idx": 0,
                "direction": 1,
                "flicker_state": self.rng.random()
            })
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]  # 0:none, 1:up, 2:down, 3:left, 4:right

        # --- Update Player Position ---
        px, py = self.player_pos
        if movement == 1: # Up (Iso: -Y)
            py -= 1
        elif movement == 2: # Down (Iso: +Y)
            py += 1
        elif movement == 3: # Left (Iso: -X)
            px -= 1
        elif movement == 4: # Right (Iso: +X)
            px += 1

        if 0 <= px < self.MAZE_WIDTH and 0 <= py < self.MAZE_HEIGHT and self.maze[px, py] != self.TILE_WALL:
            self.player_pos = [px, py]

        # --- Update Enemy Positions ---
        self.enemy_speed = min(2.0, 0.5 + 0.05 * (self.steps // 500))
        for enemy in self.enemies:
            if not enemy["path"]: continue

            target_pos = enemy["path"][enemy["path_idx"]]
            current_pos = np.array(enemy["pos"], dtype=float)
            target_pos_np = np.array(target_pos, dtype=float)

            direction_vec = target_pos_np - current_pos
            distance = np.linalg.norm(direction_vec)

            if distance < self.enemy_speed:
                enemy["pos"] = list(target_pos)
                enemy["path_idx"] += enemy["direction"]
                if not (0 <= enemy["path_idx"] < len(enemy["path"])):
                    enemy["direction"] *= -1
                    enemy["path_idx"] += 2 * enemy["direction"]
                    enemy["path_idx"] = max(0, min(len(enemy["path"]) - 1, enemy["path_idx"]))
            else:
                move_vec = (direction_vec / distance) * self.enemy_speed
                enemy["pos"] = list(current_pos + move_vec)

        # --- Check Game Events & Calculate Rewards ---
        
        # Continuous reward for being in shadow/light
        player_tile = self.maze[self.player_pos[0], self.player_pos[1]]
        if player_tile == self.TILE_SHADOW:
            reward += 0.1 # Small reward for staying in shadows
        else: # TILE_EMPTY
            reward -= 0.2 # Small penalty for being in the open
        
        # Check for reaching checkpoints
        for i, cp_pos in enumerate(self.checkpoints):
            if self.player_pos == cp_pos and i not in self.reached_checkpoints:
                self.reached_checkpoints.append(i)
                reward += 5.0
                self.score += 50
                # Sound: Checkpoint reached!

        # Check for enemy collision
        for enemy in self.enemies:
            if int(enemy["pos"][0]) == self.player_pos[0] and int(enemy["pos"][1]) == self.player_pos[1]:
                reward = -100.0
                self.score -= 100
                terminated = True
                self.game_over = True
                self.game_outcome = "DEFEAT"
                # Sound: Player caught!
                break
        
        # Check for reaching exit
        if not terminated and self.player_pos == self.exit_pos:
            reward = 100.0
            self.score += 1000
            terminated = True
            self.game_over = True
            self.game_outcome = "VICTORY"
            # Sound: Victory fanfare!

        # --- Update Step Count and Check for Max Steps ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            reward = -100.0 # Penalty for running out of time
            terminated = True
            self.game_over = True
            self.game_outcome = "DEFEAT"
            # Sound: Time's up!

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        """Renders the main game world (maze, entities)."""
        # Center camera on player
        player_screen_x, player_screen_y = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        self.camera_offset = [
            self.screen_size[0] / 2 - player_screen_x + self.camera_offset[0],
            self.screen_size[1] / 2 - player_screen_y + self.camera_offset[1]
        ]

        # Draw tiles from back to front
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                screen_x, screen_y = self._iso_to_screen(x, y)
                tile_type = self.maze[x, y]

                if tile_type == self.TILE_WALL:
                    # Simple wall block
                    points = [
                        (screen_x, screen_y - self.TILE_HEIGHT),
                        (screen_x + self.TILE_WIDTH_HALF, screen_y - self.TILE_HEIGHT_HALF),
                        (screen_x, screen_y),
                        (screen_x - self.TILE_WIDTH_HALF, screen_y - self.TILE_HEIGHT_HALF)
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_WALL, points)
                else:
                    # Floor or shadow tile
                    points = [
                        (screen_x, screen_y),
                        (screen_x + self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF),
                        (screen_x, screen_y + self.TILE_HEIGHT),
                        (screen_x - self.TILE_WIDTH_HALF, screen_y + self.TILE_HEIGHT_HALF)
                    ]
                    color = self.COLOR_SHADOW if tile_type == self.TILE_SHADOW else self.COLOR_FLOOR
                    pygame.draw.polygon(self.screen, color, points)

        # Draw checkpoints
        for i, cp_pos in enumerate(self.checkpoints):
            sx, sy = self._iso_to_screen(cp_pos[0], cp_pos[1])
            color = self.COLOR_CHECKPOINT_REACHED if i in self.reached_checkpoints else self.COLOR_CHECKPOINT
            pygame.gfxdraw.filled_circle(self.screen, sx, int(sy + self.TILE_HEIGHT_HALF), 8, color)
            pygame.gfxdraw.aacircle(self.screen, sx, int(sy + self.TILE_HEIGHT_HALF), 8, color)

        # Draw exit with glow
        sx, sy = self._iso_to_screen(self.exit_pos[0], self.exit_pos[1])
        rect = pygame.Rect(sx - 10, sy, 20, 20)
        glow_color = (*self.COLOR_EXIT, 30)
        for i in range(5, 0, -1):
             pygame.draw.rect(self.screen, glow_color, rect.inflate(i*4, i*4), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, rect, border_radius=3)

        # Draw enemies
        for enemy in self.enemies:
            sx, sy = self._iso_to_screen(enemy["pos"][0], enemy["pos"][1])
            flicker = 0.75 + 0.25 * math.sin(pygame.time.get_ticks() * 0.02 + enemy["flicker_state"] * 10)
            radius = int(8 * flicker)
            color = (int(self.COLOR_ENEMY[0]*flicker), self.COLOR_ENEMY[1], self.COLOR_ENEMY[2])
            pygame.gfxdraw.filled_circle(self.screen, sx, int(sy + self.TILE_HEIGHT_HALF), radius, color)
            pygame.gfxdraw.aacircle(self.screen, sx, int(sy + self.TILE_HEIGHT_HALF), radius, color)
            
        # Draw player
        sx, sy = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, sx, int(sy + self.TILE_HEIGHT_HALF), 6, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, sx, int(sy + self.TILE_HEIGHT_HALF), 6, self.COLOR_PLAYER)

    def _render_fog_of_war(self):
        """Renders the limited visibility effect."""
        fog_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        fog_surface.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 255))
        
        # Center of visibility on the player's screen position
        center_x, center_y = self.screen_size[0] // 2, self.screen_size[1] // 2
        
        # Create a radial gradient for a soft-edged visibility circle
        for r in range(int(self.VISIBILITY_RADIUS), 0, -2):
            alpha = 255 - int(255 * (r / self.VISIBILITY_RADIUS)**0.5)
            pygame.gfxdraw.filled_circle(fog_surface, center_x, center_y, r, (0, 0, 0, alpha))
            
        self.screen.blit(fog_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    def _render_ui(self):
        """Renders UI elements like score and game over text."""
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.screen_size[0] - steps_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            outcome_text_str = "VICTORY!" if self.game_outcome == "VICTORY" else "YOU WERE CAUGHT"
            color = (150, 255, 150) if self.game_outcome == "VICTORY" else (255, 100, 100)
            
            outcome_text = self.font_game_over.render(outcome_text_str, True, color)
            text_rect = outcome_text.get_rect(center=(self.screen_size[0] / 2, self.screen_size[1] / 2))
            self.screen.blit(outcome_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_fog_of_war()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=123)
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption("Isometric Horror Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Press R to reset
                    print("--- RESETTING ---")
                    obs, info = env.reset()
                    total_reward = 0

        # Since auto_advance is False, we only step on an action
        # For manual play, we step every frame to feel responsive
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode Finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            print("--- RESETTING ---")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(10) # Control manual play speed

    env.close()