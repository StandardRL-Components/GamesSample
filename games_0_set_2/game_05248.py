
# Generated: 2025-08-28T04:25:30.882717
# Source Brief: brief_05248.md
# Brief Index: 5248

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character (white square) one tile at a time. "
        "Avoid the ghosts!"
    )

    game_description = (
        "Navigate a spooky graveyard, collecting 5 glowing artifacts while evading patrolling ghosts. "
        "Each move is critical in this tense, turn-based survival game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.TILE_SIZE = self.HEIGHT // self.GRID_H
        self.MAX_STEPS = 1000
        self.NUM_ARTIFACTS = 5
        self.NUM_GHOSTS = 3

        # --- Colors ---
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_FLOOR = (40, 40, 60)
        self.COLOR_WALL = (10, 10, 10)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_GHOST = (220, 220, 255)
        self.COLOR_ARTIFACT = (255, 220, 0)
        self.COLOR_ARTIFACT_GLOW = (255, 220, 0, 50)
        self.COLOR_UI_TEXT = (200, 200, 200)
        self.COLOR_UI_ICON_EMPTY = (70, 70, 90)

        # --- Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_big = pygame.font.Font(None, 72)
        
        # --- Maze & Paths ---
        self.MAZE_LAYOUT = [
            "################",
            "#@.............#",
            "#.####.###.####.",
            "#.#..#.#...#..#.#",
            "#.#..#.#.###.##.#",
            "#....###.....#..#",
            "#.######.###.#.##",
            "#.#....#.#...#..#",
            "#.##########.##.#",
            "################",
        ]
        self.GHOST_PATHS = [
            # Path 1: Outer loop
            [(x, 1) for x in range(1, 15)] + [(14, y) for y in range(2, 9)] + [(x, 8) for x in range(14, 0, -1)] + [(1, y) for y in range(8, 1, -1)],
            # Path 2: Central 'S'
            [(7, 2), (7, 3), (7, 4), (6, 4), (5, 4), (5, 5), (5, 6), (6, 6), (7, 6), (7, 7), (7, 6), (6, 6), (5, 6), (5, 5), (5, 4), (6, 4), (7, 4), (7, 3)],
            # Path 3: Right side box
            [(13, 3), (12, 3), (11, 3), (11, 4), (11, 5), (12, 5), (13, 5), (13, 4)],
        ]
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = (0, 0)
        self.artifacts = []
        self.artifacts_collected = 0
        self.ghosts = []
        self.np_random = None

        self.validate_implementation()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.artifacts_collected = 0

        empty_cells = []
        for r, row in enumerate(self.MAZE_LAYOUT):
            for c, char in enumerate(row):
                if char == '.':
                    empty_cells.append((c, r))
                elif char == '@':
                    self.player_pos = (c, r)
        
        self.np_random.shuffle(empty_cells)
        artifact_indices = self.np_random.choice(len(empty_cells), self.NUM_ARTIFACTS, replace=False)
        self.artifacts = [empty_cells[i] for i in artifact_indices]

        self.ghosts = []
        for i in range(self.NUM_GHOSTS):
            path = self.GHOST_PATHS[i]
            start_index = self.np_random.integers(0, len(path))
            self.ghosts.append({
                "path": path,
                "path_index": start_index,
                "pos": path[start_index]
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.2  # Cost for taking a step
        terminated = False
        
        # --- Update Player Position ---
        px, py = self.player_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        if self.MAZE_LAYOUT[py][px] != '#':
            self.player_pos = (px, py)

        # --- Update Ghost Positions ---
        for ghost in self.ghosts:
            ghost["path_index"] = (ghost["path_index"] + 1) % len(ghost["path"])
            ghost["pos"] = ghost["path"][ghost["path_index"]]

        # --- Check Interactions ---
        # Artifact collection
        if self.player_pos in self.artifacts:
            self.artifacts.remove(self.player_pos)
            self.artifacts_collected += 1
            reward += 10
            # SFX: Collect sound

        # Ghost collision
        for ghost in self.ghosts:
            if self.player_pos == ghost["pos"]:
                terminated = True
                reward -= 50
                self.win = False
                # SFX: Player death sound
                break
        
        # --- Check Termination Conditions ---
        if self.artifacts_collected == self.NUM_ARTIFACTS:
            terminated = True
            reward += 100
            self.win = True
            # SFX: Victory fanfare
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_offset_x = (self.WIDTH - (self.GRID_W * self.TILE_SIZE)) // 2
        
        for r, row in enumerate(self.MAZE_LAYOUT):
            for c, tile in enumerate(row):
                rect = pygame.Rect(
                    grid_offset_x + c * self.TILE_SIZE,
                    r * self.TILE_SIZE,
                    self.TILE_SIZE,
                    self.TILE_SIZE
                )
                if tile == '#':
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_FLOOR, rect)

        # Render Artifacts with glow
        for x, y in self.artifacts:
            center_x = int(grid_offset_x + x * self.TILE_SIZE + self.TILE_SIZE / 2)
            center_y = int(y * self.TILE_SIZE + self.TILE_SIZE / 2)
            glow_radius = int(self.TILE_SIZE * 0.7)
            
            # Use a temporary surface for the glow alpha effect
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_ARTIFACT_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (center_x - glow_radius, center_y - glow_radius))

            pygame.draw.rect(self.screen, self.COLOR_ARTIFACT, pygame.Rect(
                grid_offset_x + x * self.TILE_SIZE + self.TILE_SIZE * 0.2,
                y * self.TILE_SIZE + self.TILE_SIZE * 0.2,
                self.TILE_SIZE * 0.6,
                self.TILE_SIZE * 0.6
            ))

        # Render Ghosts with flicker
        for ghost in self.ghosts:
            x, y = ghost["pos"]
            center_x = int(grid_offset_x + x * self.TILE_SIZE + self.TILE_SIZE / 2)
            center_y = int(y * self.TILE_SIZE + self.TILE_SIZE / 2)
            radius = int(self.TILE_SIZE * 0.4)
            
            # Flicker effect by varying alpha
            alpha = 128 + 64 * math.sin(pygame.time.get_ticks() * 0.01 + id(ghost))
            
            # Using gfxdraw for anti-aliased, alpha-blended circles
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_GHOST + (int(alpha),))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_GHOST + (int(alpha),))

        # Render Player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            grid_offset_x + px * self.TILE_SIZE + self.TILE_SIZE * 0.1,
            py * self.TILE_SIZE + self.TILE_SIZE * 0.1,
            self.TILE_SIZE * 0.8,
            self.TILE_SIZE * 0.8
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, player_rect, 2) # Border

    def _render_ui(self):
        # Render Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Render Artifacts collected
        for i in range(self.NUM_ARTIFACTS):
            center = (240 + i * 30, 30)
            color = self.COLOR_ARTIFACT if i < self.artifacts_collected else self.COLOR_UI_ICON_EMPTY
            pygame.draw.circle(self.screen, color, center, 10)
            pygame.draw.circle(self.screen, self.COLOR_BG, center, 10, 2)

        # Render Game Over/Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_ARTIFACT if self.win else (200, 50, 50)
            
            end_text = self.font_big.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "artifacts_collected": self.artifacts_collected,
        }

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Graveyard Ghost")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_q: # Quit
                    running = False
        
        if not terminated:
            # Since auto_advance is False, we only step when there's an action
            # For human play, we step on every key press
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # If the game ends, wait for R to reset or Q to quit
        if terminated:
            print("Game Over! Press 'R' to reset or 'Q' to quit.")

    env.close()