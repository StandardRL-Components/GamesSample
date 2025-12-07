import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:29:47.006022
# Source Brief: brief_00454.md
# Brief Index: 454
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
        "Navigate a bouncing square through a fluctuating maze. Avoid deadly red walls and reach the yellow exit before time runs out."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to change the direction of your bouncing square."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 5000 # Increased to allow for more complex levels

    # --- Colors ---
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 40)
    COLOR_WALL_SAFE = (100, 200, 255)
    COLOR_WALL_DEADLY = (255, 50, 50)
    COLOR_EXIT = (255, 220, 0)
    COLOR_EXIT_GLOW = (255, 220, 0, 60)
    COLOR_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_game_over = pygame.font.SysFont('Consolas', 60, bold=True)
        self._background_surf = self._create_gradient_background()

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.timer = 0.0

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = 0

        self.walls = []
        self.exit_pos = pygame.Vector2(0, 0)
        self.exit_base_radius = 0

        self.wall_fluctuation_period = 0.0
        self.wall_fluctuation_amplitude = 0.0
        
        # --- Initialize state variables and validate ---
        # self.reset() # reset() is called by the environment runner, no need to call it here
        # self.validate_implementation() # Validation is for debugging, not production code


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = self.game_over

        if not terminated:
            movement = action[0]
            # space_held and shift_held are unused as per brief
            
            self._update_player_direction(movement)
            self._update_physics()
            
            reward += 0.01  # Reward for surviving

            # Check for level completion
            if self.player_pos.distance_to(self.exit_pos) < self.exit_base_radius + self.player_size / 2:
                # SFX: Level Complete
                self.level += 1
                self.score += 1
                if self.level > 5:
                    reward += 100  # Victory
                    terminated = True
                    self.game_over = True
                else:
                    reward += 1 # Progress
                    self._setup_level()
            
            # Check for termination conditions
            if self.timer <= 0:
                # SFX: Timeout
                reward = -100
                terminated = True
                self.game_over = True

            if self.steps >= self.MAX_STEPS:
                terminated = True

            if self.game_over and not terminated: # game_over set by collision
                # SFX: Player Death
                reward = -100
                terminated = True

        self.steps += 1
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _setup_level(self):
        """Initializes all game elements for the current level."""
        self.timer = 60.0
        self.wall_fluctuation_period = max(3.0, 5.0 - (self.level - 1) * 0.4)
        
        # Maze dimensions scale with level
        cols = 8 + self.level * 2
        rows = 5 + self.level
        
        base_cell_w = self.SCREEN_WIDTH / (cols + 1)
        base_cell_h = self.SCREEN_HEIGHT / (rows + 1)
        self.wall_fluctuation_amplitude = min(base_cell_w, base_cell_h) * 0.25
        
        self.player_size = min(base_cell_w, base_cell_h) * 0.6
        self.exit_base_radius = self.player_size
        
        self._generate_maze(cols, rows, base_cell_w, base_cell_h)

        # Set player initial state
        self.player_pos = pygame.Vector2(self.start_pos)
        initial_speed = 2.5 + self.level * 0.2
        self.player_vel = pygame.Vector2(self.np_random.choice([-1, 1]), self.np_random.choice([-1, 1])) * initial_speed

    def _generate_maze(self, cols, rows, cell_w, cell_h):
        """Generates a maze using Prim's algorithm and converts it to wall rects."""
        grid = [[{'N': True, 'E': True, 'S': True, 'W': True, 'visited': False} for _ in range(cols)] for _ in range(rows)]
        
        start_cell = (self.np_random.integers(0, rows), self.np_random.integers(0, cols))
        grid[start_cell[0]][start_cell[1]]['visited'] = True
        
        wall_list = []
        for d, move in [('N', (-1, 0)), ('E', (0, 1)), ('S', (1, 0)), ('W', (0, -1))]:
            if 0 <= start_cell[0] + move[0] < rows and 0 <= start_cell[1] + move[1] < cols:
                wall_list.append((start_cell, d))

        end_cell = start_cell
        while wall_list:
            wall_idx = self.np_random.integers(0, len(wall_list))
            (r, c), direction = wall_list.pop(wall_idx)
            
            if direction == 'N': dr, dc, opposite = -1, 0, 'S'
            elif direction == 'E': dr, dc, opposite = 0, 1, 'W'
            elif direction == 'S': dr, dc, opposite = 1, 0, 'N'
            else: dr, dc, opposite = 0, -1, 'E'
            
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc]['visited']:
                grid[r][c][direction] = False
                grid[nr][nc][opposite] = False
                grid[nr][nc]['visited'] = True
                end_cell = (nr, nc) # Furthest cell becomes the end
                
                for nd, move in [('N', (-1, 0)), ('E', (0, 1)), ('S', (1, 0)), ('W', (0, -1))]:
                    if 0 <= nr + move[0] < rows and 0 <= nc + move[1] < cols:
                        wall_list.append(((nr, nc), nd))
        
        # Convert grid to wall rects
        self.walls = []
        margin_x = (self.SCREEN_WIDTH - cols * cell_w) / 2
        margin_y = (self.SCREEN_HEIGHT - rows * cell_h) / 2
        
        for r in range(rows):
            for c in range(cols):
                x, y = margin_x + c * cell_w, margin_y + r * cell_h
                if grid[r][c]['N']:
                    self.walls.append({'base_rect': pygame.Rect(x, y, cell_w, 4), 'is_deadly': False})
                if grid[r][c]['W']:
                    self.walls.append({'base_rect': pygame.Rect(x, y, 4, cell_h), 'is_deadly': False})
                if c == cols - 1 and grid[r][c]['E']:
                    self.walls.append({'base_rect': pygame.Rect(x + cell_w, y, 4, cell_h), 'is_deadly': False})
                if r == rows - 1 and grid[r][c]['S']:
                    self.walls.append({'base_rect': pygame.Rect(x, y + cell_h, cell_w, 4), 'is_deadly': False})

        self.start_pos = (margin_x + start_cell[1] * cell_w + cell_w / 2, margin_y + start_cell[0] * cell_h + cell_h / 2)
        self.exit_pos = pygame.Vector2(margin_x + end_cell[1] * cell_w + cell_w / 2, margin_y + end_cell[0] * cell_h + cell_h / 2)
        
        # Make 10% of walls deadly, avoiding start/exit neighbors
        num_deadly = int(len(self.walls) * 0.1)
        candidate_indices = [i for i, wall in enumerate(self.walls) if pygame.Vector2(wall['base_rect'].center).distance_to(self.start_pos) > cell_w * 2 and pygame.Vector2(wall['base_rect'].center).distance_to(self.exit_pos) > cell_w * 2]
        if len(candidate_indices) > num_deadly:
            deadly_indices = self.np_random.choice(candidate_indices, num_deadly, replace=False)
            for i in deadly_indices:
                self.walls[i]['is_deadly'] = True

    def _update_player_direction(self, movement):
        if movement == 0: # none
            return
        # SFX: Direction Change
        if movement == 1: # up
            self.player_vel.y = -abs(self.player_vel.y)
        elif movement == 2: # down
            self.player_vel.y = abs(self.player_vel.y)
        elif movement == 3: # left
            self.player_vel.x = -abs(self.player_vel.x)
        elif movement == 4: # right
            self.player_vel.x = abs(self.player_vel.x)

    def _update_physics(self):
        self.timer -= 1.0 / self.FPS
        self.player_pos += self.player_vel

        player_rect = pygame.Rect(self.player_pos.x - self.player_size / 2, self.player_pos.y - self.player_size / 2, self.player_size, self.player_size)

        # Screen boundary collisions
        if player_rect.left < 0 or player_rect.right > self.SCREEN_WIDTH:
            self.player_vel.x *= -1
            player_rect.left = max(0, player_rect.left)
            player_rect.right = min(self.SCREEN_WIDTH, player_rect.right)
            self.player_pos.x = player_rect.centerx
            # SFX: Bounce
        if player_rect.top < 0 or player_rect.bottom > self.SCREEN_HEIGHT:
            self.player_vel.y *= -1
            player_rect.top = max(0, player_rect.top)
            player_rect.bottom = min(self.SCREEN_HEIGHT, player_rect.bottom)
            self.player_pos.y = player_rect.centery
            # SFX: Bounce

        # Wall collisions
        time_phase = (self.steps / self.FPS) * (2 * math.pi / self.wall_fluctuation_period)
        fluctuation = self.wall_fluctuation_amplitude * math.sin(time_phase)
        
        for wall in self.walls:
            w_rect = wall['base_rect'].copy()
            if w_rect.width > w_rect.height: # Horizontal wall
                w_rect.h += fluctuation
            else: # Vertical wall
                w_rect.w += fluctuation
            w_rect.center = wall['base_rect'].center

            if player_rect.colliderect(w_rect):
                if wall['is_deadly']:
                    self.game_over = True
                    return

                # Bounce logic
                overlap_x = min(player_rect.right, w_rect.right) - max(player_rect.left, w_rect.left)
                overlap_y = min(player_rect.bottom, w_rect.bottom) - max(player_rect.top, w_rect.top)

                if overlap_x < overlap_y:
                    self.player_vel.x *= -1
                    self.player_pos.x += self.player_vel.x / self.FPS # Push out
                else:
                    self.player_vel.y *= -1
                    self.player_pos.y += self.player_vel.y / self.FPS # Push out
                # SFX: Bounce
                return # Process one collision per frame

    def _get_observation(self):
        self.screen.blit(self._background_surf, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Calculate sinusoidal fluctuations for this frame
        time_phase = (self.steps / self.FPS) * (2 * math.pi / self.wall_fluctuation_period)
        exit_pulsation = 5 * math.sin(time_phase * 2)

        # Render Exit
        exit_radius = int(max(5, self.exit_base_radius + exit_pulsation))
        pygame.gfxdraw.filled_circle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), exit_radius + 10, self.COLOR_EXIT_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), exit_radius, self.COLOR_EXIT)
        pygame.gfxdraw.filled_circle(self.screen, int(self.exit_pos.x), int(self.exit_pos.y), exit_radius, self.COLOR_EXIT)

        # Render Walls
        for wall in self.walls:
            color = self.COLOR_WALL_DEADLY if wall['is_deadly'] else self.COLOR_WALL_SAFE
            w_rect = wall['base_rect'].copy()
            
            current_fluctuation = self.wall_fluctuation_amplitude * math.sin(time_phase + (wall['base_rect'].x + wall['base_rect'].y) * 0.1) # Add phase shift for variety
            
            if w_rect.width > w_rect.height:
                w_rect.height = max(1, w_rect.height + current_fluctuation)
            else:
                w_rect.width = max(1, w_rect.width + current_fluctuation)
            w_rect.center = wall['base_rect'].center
            pygame.draw.rect(self.screen, color, w_rect, border_radius=2)
        
        # Render Player
        player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        
        glow_size = int(self.player_size * 1.8)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=int(glow_size*0.2))
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size/2, player_rect.centery - glow_size/2))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=int(self.player_size*0.2))

    def _render_ui(self):
        level_text = self.font_ui.render(f"Level: {self.level}/5", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        timer_text = self.font_ui.render(f"Time: {max(0, self.timer):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.level > 5 else "GAME OVER"
            color = self.COLOR_EXIT if self.level > 5 else self.COLOR_WALL_DEADLY
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer
        }

    def _create_gradient_background(self):
        """Creates a pre-rendered surface with the background gradient."""
        surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(surf, color, (0, y), (self.SCREEN_WIDTH, y))
        return surf

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Recursive Maze")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            # Wait for a moment on the game over screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render ---
        # The observation is already a rendered image, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()