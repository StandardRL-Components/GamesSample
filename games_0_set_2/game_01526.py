
# Generated: 2025-08-27T17:25:42.928960
# Source Brief: brief_01526.md
# Brief Index: 1526

        
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

    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place a mine. Shift to detonate all placed mines."
    )

    game_description = (
        "A strategic puzzle game. Place mines on the grid to destroy all rocks within the move limit. "
        "Each mine placement costs one move. Detonating is free. Plan your chain reactions carefully!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game Constants ---
        self._define_constants()
        
        # --- State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.moves_remaining = None
        self.rocks_remaining = None
        self.initial_rock_count = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_status = None
        self.prev_action = None
        self.move_timer = None
        self.explosions = None
        self.particles = None
        self.last_destroyed_count = None
        
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()

    def _define_constants(self):
        """Define all magic numbers and constants for the game."""
        self.GRID_W, self.GRID_H = 20, 12
        self.CELL_SIZE = 28
        self.GRID_MARGIN_X = (self.screen_width - (self.GRID_W * self.CELL_SIZE)) // 2
        self.GRID_MARGIN_Y = (self.screen_height - (self.GRID_H * self.CELL_SIZE) + 40) // 2

        # Cell types
        self.EMPTY, self.ROCK, self.MINE = 0, 1, 2
        
        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 52, 64)
        self.COLOR_ROCK = (108, 117, 125)
        self.COLOR_ROCK_BORDER = (80, 90, 100)
        self.COLOR_MINE = (255, 77, 77)
        self.COLOR_MINE_PULSE = (255, 150, 150)
        self.COLOR_CURSOR = (46, 204, 113)
        self.COLOR_EXPLOSION = (252, 196, 25)
        self.COLOR_PARTICLE = (220, 220, 220)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SUCCESS = (46, 204, 113)
        self.COLOR_TEXT_FAIL = (255, 77, 77)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

        # Game parameters
        self.INITIAL_MOVES = 15
        self.INITIAL_ROCKS = 10
        self.MAX_STEPS = 30 * 60 # 60 seconds at 30fps
        self.MOVE_COOLDOWN = 0.12 # seconds

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.full((self.GRID_W, self.GRID_H), self.EMPTY, dtype=np.uint8)
        self.cursor_pos = np.array([self.GRID_W // 2, self.GRID_H // 2])
        
        self.moves_remaining = self.INITIAL_MOVES
        self.initial_rock_count = self.INITIAL_ROCKS
        
        # Place rocks randomly
        empty_cells = list(np.ndindex(self.grid.shape))
        self.np_random.shuffle(empty_cells)
        for i in range(self.initial_rock_count):
            if not empty_cells: break
            x, y = empty_cells.pop()
            self.grid[x, y] = self.ROCK
        self.rocks_remaining = np.sum(self.grid == self.ROCK)

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_status = None # "WIN" or "LOSS"
        
        self.prev_action = np.array([0, 0, 0])
        self.move_timer = 0
        
        self.explosions = []
        self.particles = []
        self.last_destroyed_count = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        dt = self.clock.tick(30) / 1000.0
        reward = 0

        self._update_animations(dt)
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_action[1]
        shift_press = shift_held and not self.prev_action[2]

        # --- Handle Input ---
        self.move_timer -= dt
        if movement != 0 and self.move_timer <= 0:
            self._move_cursor(movement)
            self.move_timer = self.MOVE_COOLDOWN

        if space_press:
            self._place_mine()

        if shift_press:
            detonation_reward, rocks_destroyed = self._handle_detonation()
            reward += detonation_reward
            self.last_destroyed_count = rocks_destroyed
            
        self.prev_action = np.array([movement, space_held, shift_held])
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win_status == "WIN":
                reward += 100
            elif self.win_status == "LOSS":
                reward -= 100

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

    def _place_mine(self):
        x, y = self.cursor_pos
        if self.moves_remaining > 0 and self.grid[x, y] == self.EMPTY:
            self.grid[x, y] = self.MINE
            self.moves_remaining -= 1
            # sfx: place_mine.wav

    def _handle_detonation(self):
        mines = np.argwhere(self.grid == self.MINE)
        if len(mines) == 0:
            return 0, 0

        # sfx: detonate_trigger.wav
        cells_to_destroy = set()
        for x, y in mines:
            # Create explosion animation
            px, py = self._grid_to_pixel(x, y)
            self.explosions.append({"pos": (px, py), "radius": 0, "max_radius": self.CELL_SIZE * 1.5, "life": 0.5})
            
            # Add 3x3 area to destruction set
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                        cells_to_destroy.add((nx, ny))

        rocks_destroyed = 0
        for x, y in cells_to_destroy:
            if self.grid[x, y] == self.ROCK:
                self.grid[x, y] = self.EMPTY
                rocks_destroyed += 1
                self._spawn_rock_particles(x, y)
                # sfx: explosion_small.wav

        self.grid[self.grid == self.MINE] = self.EMPTY
        self.rocks_remaining -= rocks_destroyed
        
        reward = rocks_destroyed * 1.0
        return reward, rocks_destroyed

    def _check_termination(self):
        if self.rocks_remaining <= 0:
            self.win_status = "WIN"
            return True
        
        num_mines_on_board = np.sum(self.grid == self.MINE)
        if self.moves_remaining <= 0 and num_mines_on_board == 0:
            self.win_status = "LOSS"
            return True

        if self.steps >= self.MAX_STEPS:
            self.win_status = "LOSS"
            return True
            
        return False

    def _update_animations(self, dt):
        # Update explosions
        for explosion in self.explosions[:]:
            explosion["life"] -= dt
            explosion["radius"] += (explosion["max_radius"] / 0.5) * dt
            if explosion["life"] <= 0:
                self.explosions.remove(explosion)

        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"] * dt
            p["life"] -= dt
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_rock_particles(self, grid_x, grid_y):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(px, py),
                "vel": vel,
                "life": random.uniform(0.3, 0.7),
                "size": random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, x, y):
        px = self.GRID_MARGIN_X + x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_MARGIN_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2
        return int(px), int(py)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_W + 1):
            start_pos = (self.GRID_MARGIN_X + x * self.CELL_SIZE, self.GRID_MARGIN_Y)
            end_pos = (self.GRID_MARGIN_X + x * self.CELL_SIZE, self.GRID_MARGIN_Y + self.GRID_H * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_H + 1):
            start_pos = (self.GRID_MARGIN_X, self.GRID_MARGIN_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_MARGIN_X + self.GRID_W * self.CELL_SIZE, self.GRID_MARGIN_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw grid contents
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                px, py = self._grid_to_pixel(x, y)
                cell_rect = pygame.Rect(px - self.CELL_SIZE // 2, py - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)
                
                if self.grid[x, y] == self.ROCK:
                    pygame.draw.rect(self.screen, self.COLOR_ROCK, cell_rect.inflate(-6, -6), border_radius=4)
                    pygame.draw.rect(self.screen, self.COLOR_ROCK_BORDER, cell_rect.inflate(-6, -6), 2, border_radius=4)
                elif self.grid[x, y] == self.MINE:
                    rad = int(self.CELL_SIZE * 0.35)
                    pulse_rad = rad + 2 * math.sin(self.steps * 0.2)
                    pygame.gfxdraw.filled_circle(self.screen, px, py, rad, self.COLOR_MINE)
                    pygame.gfxdraw.aacircle(self.screen, px, py, int(pulse_rad), self.COLOR_MINE_PULSE)

        # Draw explosions
        for explosion in self.explosions:
            alpha = max(0, min(255, int(255 * (explosion["life"] / 0.5))))
            color = (*self.COLOR_EXPLOSION, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(explosion["pos"][0]), int(explosion["pos"][1]), int(explosion["radius"]), color)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 0.7))))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.draw.circle(self.screen, color, p["pos"], p["size"])
            
        # Draw cursor
        cx, cy = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        size = self.CELL_SIZE // 2
        rect = pygame.Rect(cx - size, cy - size, size * 2, size * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=3)

    def _render_ui(self):
        # Helper to draw text
        def draw_text(text, font, color, pos, align="topleft"):
            surface = font.render(text, True, color)
            rect = surface.get_rect()
            setattr(rect, align, pos)
            self.screen.blit(surface, rect)

        # Top-left info
        draw_text(f"Moves: {self.moves_remaining}", self.font_small, self.COLOR_TEXT, (15, 15))
        draw_text(f"Rocks: {self.rocks_remaining}", self.font_small, self.COLOR_TEXT, (15, 35))
        
        # Top-right info
        draw_text(f"Score: {int(self.score)}", self.font_small, self.COLOR_TEXT, (self.screen_width - 15, 15), "topright")
        
        # Bottom center status
        if self.last_destroyed_count > 0:
            status_text = f"Detonation destroyed {self.last_destroyed_count} rocks!"
            draw_text(status_text, self.font_small, self.COLOR_TEXT_SUCCESS, (self.screen_width / 2, self.screen_height - 25), "midbottom")
        else:
            draw_text(self.user_guide, self.font_small, self.COLOR_TEXT, (self.screen_width / 2, self.screen_height - 25), "midbottom")
        
        # Game Over Overlay
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            if self.win_status == "WIN":
                msg = "LEVEL CLEARED!"
                color = self.COLOR_TEXT_SUCCESS
            else:
                msg = "GAME OVER"
                color = self.COLOR_TEXT_FAIL
            
            draw_text(msg, self.font_large, color, (self.screen_width / 2, self.screen_height / 2), "center")


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "rocks_remaining": self.rocks_remaining,
            "cursor_pos": list(self.cursor_pos),
            "win_status": self.win_status
        }
        
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Mine Grid")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Get human input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Score: {info['score']}, Status: {info['win_status']}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        # --- Render the observation ---
        # The observation is already a rendered frame, we just need to display it.
        # It's (H, W, C), but pygame blit needs (W, H) surface.
        # env._get_observation() already created the frame on env.screen
        screen.blit(env.screen, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()