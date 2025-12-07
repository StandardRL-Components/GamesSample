
# Generated: 2025-08-27T14:53:40.780890
# Source Brief: brief_00822.md
# Brief Index: 822

        
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
    """
    A Gymnasium environment for a maze-based gem collection game.

    The agent navigates a procedurally generated maze to collect all gems
    within a limited number of moves. The goal is to find the most
    efficient path to maximize the score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Collect all the blue gems before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A procedural maze puzzle. Collect all 25 gems in 50 moves or "
        "less to win. Plan your path carefully to achieve a high score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAZE_WIDTH = 31  # Must be odd
    MAZE_HEIGHT = 19 # Must be odd
    NUM_GEMS = 25
    MAX_MOVES = 50
    
    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_WALL = (80, 90, 100)
    COLOR_PATH = (45, 50, 60)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_GLOW = (255, 200, 0, 50)
    COLOR_GEM = (0, 180, 255)
    COLOR_GEM_GLOW = (0, 180, 255, 70)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Game state variables (initialized in reset)
        self.maze = None
        self.player_pos = None
        self.gems = None
        self.moves_remaining = 0
        self.gems_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_action_was_move = False
        self.particles = []

        # Validate implementation against requirements
        self.validate_implementation()

    def _generate_maze(self, width, height):
        """Generates a maze using randomized Depth-First Search."""
        maze = np.ones((height, width), dtype=np.uint8) # 1 = wall
        start_x, start_y = (self.np_random.integers(0, width // 2) * 2 + 1,
                            self.np_random.integers(0, height // 2) * 2 + 1)
        
        maze[start_y, start_x] = 0 # 0 = path
        stack = [(start_x, start_y)]
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < width and 0 < ny < height and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                mx, my = (cx + nx) // 2, (cy + ny) // 2
                maze[ny, nx] = 0
                maze[my, mx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.maze = self._generate_maze(self.MAZE_WIDTH, self.MAZE_HEIGHT)
        
        path_cells = np.argwhere(self.maze == 0)
        
        # Place player
        player_idx = self.np_random.integers(0, len(path_cells))
        self.player_pos = tuple(path_cells[player_idx][::-1]) # (x, y)
        path_cells = np.delete(path_cells, player_idx, axis=0)
        
        # Place gems
        gem_indices = self.np_random.choice(len(path_cells), self.NUM_GEMS, replace=False)
        self.gems = [tuple(cell[::-1]) for cell in path_cells[gem_indices]]

        # Initialize state
        self.moves_remaining = self.MAX_MOVES
        self.gems_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_action_was_move = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0
        terminated = False
        
        moved = False
        if movement != 0: # 0 is no-op
            self.moves_remaining -= 1
            
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1 # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1 # Right
            
            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy

            # Check for valid move (within bounds and not a wall)
            if 0 <= new_y < self.MAZE_HEIGHT and 0 <= new_x < self.MAZE_WIDTH and self.maze[new_y, new_x] == 0:
                self.player_pos = (new_x, new_y)
                moved = True
        
        # Reward calculation
        if moved:
            reward -= 0.2  # Cost for moving
            if self.player_pos in self.gems:
                # Gem collected
                self.gems.remove(self.player_pos)
                self.gems_collected += 1
                reward += 1.2  # Net +1.0 for gem collection
                self._create_particles(self.player_pos, self.COLOR_GEM, 20)
                # Sfx: gem collect sound

                if self.gems_collected == self.NUM_GEMS:
                    reward += 10 # Bonus for final gem
            self.last_action_was_move = True
        else:
            self.last_action_was_move = False


        # Update score
        self.score += reward
        self.steps += 1

        # Check for termination conditions
        if self.gems_collected == self.NUM_GEMS:
            reward += 50  # Victory bonus
            self.score += 50
            terminated = True
            self.game_over = True
        elif self.moves_remaining <= 0:
            reward -= 50  # Penalty for running out of moves
            self.score -= 50
            terminated = True
            self.game_over = True

        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _create_particles(self, pos, color, count):
        px, py = self._grid_to_pixel(pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

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
            "moves_remaining": self.moves_remaining,
            "gems_collected": self.gems_collected,
        }

    def _grid_to_pixel(self, grid_pos):
        cell_w = self.SCREEN_WIDTH / self.MAZE_WIDTH
        cell_h = self.SCREEN_HEIGHT / self.MAZE_HEIGHT
        px = int(grid_pos[0] * cell_w + cell_w / 2)
        py = int(grid_pos[1] * cell_h + cell_h / 2)
        return px, py

    def _render_game(self):
        cell_w = self.SCREEN_WIDTH / self.MAZE_WIDTH
        cell_h = self.SCREEN_HEIGHT / self.MAZE_HEIGHT

        # Draw maze
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                rect = pygame.Rect(x * cell_w, y * cell_h, math.ceil(cell_w), math.ceil(cell_h))
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

        # Draw gems
        gem_radius = int(min(cell_w, cell_h) * 0.25)
        glow_radius = int(gem_radius * 2.5)
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        
        for gem_pos in self.gems:
            px, py = self._grid_to_pixel(gem_pos)
            
            # Pulsing glow
            current_glow_radius = int(glow_radius * (0.8 + 0.2 * pulse))
            glow_surf = pygame.Surface((current_glow_radius * 2, current_glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_GEM_GLOW, (current_glow_radius, current_glow_radius), current_glow_radius)
            self.screen.blit(glow_surf, (px - current_glow_radius, py - current_glow_radius))
            
            pygame.gfxdraw.aacircle(self.screen, px, py, gem_radius, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, px, py, gem_radius, self.COLOR_GEM)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*p['color'], alpha)
            size = max(1, int(4 * (p['life'] / 30.0)))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size,size), size)
            self.screen.blit(temp_surf, (int(p['x']-size), int(p['y']-size)))

        # Draw player
        player_size = int(min(cell_w, cell_h) * 0.7)
        px, py = self._grid_to_pixel(self.player_pos)
        
        # Player glow
        player_glow_radius = int(player_size * 1.5)
        glow_surf = pygame.Surface((player_glow_radius * 2, player_glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (player_glow_radius, player_glow_radius), player_glow_radius)
        self.screen.blit(glow_surf, (px - player_glow_radius, py - player_glow_radius))
        
        player_rect = pygame.Rect(px - player_size // 2, py - player_size // 2, player_size, player_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves: {self.moves_remaining}"
        self._render_text(moves_text, self.font_small, self.COLOR_TEXT, (10, 10))

        # Gems collected
        gems_text = f"Gems: {self.gems_collected}/{self.NUM_GEMS}"
        text_width = self.font_small.size(gems_text)[0]
        self._render_text(gems_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))

        # Score
        score_text = f"Score: {self.score:.1f}"
        self._render_text(score_text, self.font_small, self.COLOR_TEXT, (10, self.SCREEN_HEIGHT - 30))
        
        # Game Over / Win message
        if self.game_over:
            if self.gems_collected == self.NUM_GEMS:
                msg = "LEVEL COMPLETE!"
                color = self.COLOR_GEM
            else:
                msg = "OUT OF MOVES"
                color = (200, 50, 50)
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect.topleft)
            
            # Draw shadow and text
            shadow_surf = self.font_large.render(msg, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (after a reset)
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    # This allows a human to play the game.
    
    # Re-initialize pygame with a display for manual play
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Gem Collector")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    terminated = False
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("Press R to reset, Q to quit.")
    print("="*30 + "\n")

    running = True
    while running:
        # Default action is no-op
        action = np.array([0, 0, 0]) 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    print("--- Environment Reset ---")
                
                # Only register actions if the game is not over
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        # If an action was chosen, step the environment
        if not terminated and action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, "
                  f"Moves Left: {info['moves_remaining']}, Terminated: {terminated}")

        # Render the observation to the screen
        # Need to transpose back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()
    print("Game exited.")