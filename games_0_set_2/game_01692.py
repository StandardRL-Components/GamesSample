
# Generated: 2025-08-27T17:57:51.964074
# Source Brief: brief_01692.md
# Brief Index: 1692

        
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
        "Controls: Use arrow keys to move the selector. Press Space to clear a crystal group. Reach the portal before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear a path through the crystal cavern by matching adjacent gems. Plan your moves carefully to reach the glowing exit portal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = 40
        self.MAX_MOVES = 15
        self.MAX_STEPS = 1000
        self.NUM_CRYSTAL_TYPES = 4

        # Colors
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_GRID = (50, 40, 90)
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.COLOR_CURSOR = (255, 255, 255)
        self.CRYSTAL_COLORS = [
            (0, 0, 0),  # 0: Empty
            (0, 220, 220),   # 1: Cyan
            (255, 50, 150),  # 2: Magenta
            (255, 220, 0),   # 3: Yellow
            (50, 255, 50),   # 4: Green
        ]
        self.CRYSTAL_BORDERS = [
            (0, 0, 0),
            (0, 150, 150),
            (180, 20, 100),
            (180, 150, 0),
            (20, 180, 20),
        ]
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables to None, they will be set in reset()
        self.grid = None
        self.cursor_pos = None
        self.exit_pos = None
        self.moves_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = []
        self.np_random = None
        self.pulse_timer = 0

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.particles = []
        
        # Generate grid
        self.grid = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Set cursor and exit positions
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1]
        self.exit_pos = [self.GRID_WIDTH // 2, 0]
        
        # Ensure exit and start are clear
        self.grid[self.exit_pos[0], self.exit_pos[1]] = 0 # Exit is an empty space
        self.grid[self.cursor_pos[0], self.cursor_pos[1]] = 0 # Start is an empty space

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        self.steps += 1
        self.pulse_timer = (self.pulse_timer + 0.2) % (2 * math.pi)

        # 1. Handle cursor movement
        prev_cursor_pos = list(self.cursor_pos)
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Clamp cursor to grid boundaries
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # 2. Handle selection action
        if space_pressed:
            x, y = self.cursor_pos
            crystal_type = self.grid[x, y]
            
            if crystal_type > 0: # Cannot select empty space
                self.moves_remaining -= 1
                
                # Find matching group
                group = self._find_matching_group(x, y)
                
                if len(group) > 1:
                    # Successful match
                    cleared_count = len(group)
                    reward += cleared_count
                    self.score += cleared_count
                    # SFX: Crystal shatter
                    for gx, gy in group:
                        self.grid[gx, gy] = 0
                        self._create_particles(gx, gy, self.CRYSTAL_COLORS[crystal_type])
                else:
                    # Wasted move (clearing a single crystal)
                    reward -= 0.2
                    self.score -= 1 # Penalty
                    self.grid[x, y] = 0
                    # SFX: Dull thud
                    self._create_particles(x, y, (100, 100, 100))

                self._apply_gravity()
        
        # 3. Update game state
        self._update_particles()
        
        # 4. Check for termination conditions
        # Win condition: cursor is on the exit tile and the path is clear
        if self.cursor_pos == self.exit_pos:
            if self.grid[self.exit_pos[0], self.exit_pos[1]] == 0:
                reward += 10
                self.score += 50 # Bonus score for winning
                terminated = True
                self.game_over = True
                # SFX: Victory fanfare
        
        # Lose condition: out of moves
        if self.moves_remaining <= 0 and not terminated:
            terminated = True
            self.game_over = True
            # SFX: Failure sound
        
        # Lose condition: max steps reached
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _find_matching_group(self, start_x, start_y):
        target_type = self.grid[start_x, start_y]
        if target_type == 0:
            return []
        
        q = [(start_x, start_y)]
        visited = set(q)
        group = []
        
        while q:
            x, y = q.pop(0)
            group.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_type:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != empty_row:
                        self.grid[x, empty_row] = self.grid[x, y]
                        self.grid[x, y] = 0
                    empty_row -= 1
            # Fill empty spaces at the top
            for y in range(empty_row, -1, -1):
                self.grid[x, y] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)

    def _create_particles(self, grid_x, grid_y, color):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        px += self.CELL_SIZE // 2
        py += self.CELL_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, x, y):
        return x * self.CELL_SIZE, y * self.CELL_SIZE

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)
            
        # Draw exit portal
        ex, ey = self._grid_to_pixel(self.exit_pos[0], self.exit_pos[1])
        center = (ex + self.CELL_SIZE // 2, ey + self.CELL_SIZE // 2)
        radius = self.CELL_SIZE // 2.5
        pulse = (math.sin(self.pulse_timer * 1.5) + 1) / 2 # 0 to 1
        
        # Glowing effect
        for i in range(4, 0, -1):
            alpha = 80 - i * 15
            glow_radius = int(radius + pulse * 8 * (i/4))
            color = (200, 220, 255, alpha)
            s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (center[0] - glow_radius, center[1] - glow_radius))
        
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(radius), (255, 255, 255))
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(radius), (255, 255, 255))

        # Draw crystals
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                crystal_type = self.grid[x, y]
                if crystal_type > 0:
                    self._draw_crystal(x, y, crystal_type)
                    
        # Draw cursor
        cx, cy = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        pulse_alpha = 100 + 100 * (math.sin(self.pulse_timer) + 1) / 2
        cursor_color = (*self.COLOR_CURSOR, pulse_alpha)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, cursor_color, (2, 2, self.CELL_SIZE-4, self.CELL_SIZE-4), 3, border_radius=4)
        self.screen.blit(s, (cx, cy))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 6.4))) # Fade out
            color = (*p['color'], alpha)
            size = max(1, int(p['life'] / 8))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (p['pos'][0]-size, p['pos'][1]-size))
            
    def _draw_crystal(self, grid_x, grid_y, crystal_type):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        center_x = px + self.CELL_SIZE // 2
        center_y = py + self.CELL_SIZE // 2
        radius = self.CELL_SIZE * 0.4
        
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((int(x), int(y)))
            
        fill_color = self.CRYSTAL_COLORS[crystal_type]
        border_color = self.CRYSTAL_BORDERS[crystal_type]

        pygame.gfxdraw.aapolygon(self.screen, points, border_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, fill_color)

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.cursor_pos == self.exit_pos else "GAME OVER"
            end_text = self.font_large.render(message, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": list(self.cursor_pos),
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
        
        # Test observation space (requires a reset first)
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit on 'q'
                    running = False

        # Only step if an action is taken, as auto_advance is False
        if movement != 0 or space != 0 or shift != 0:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                # Optional: auto-reset after a delay
                # pygame.time.wait(2000)
                # obs, info = env.reset()

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS

    env.close()