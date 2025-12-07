
# Generated: 2025-08-27T20:29:05.058999
# Source Brief: brief_02476.md
# Brief Index: 2476

        
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

    # User-facing control string
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate clockwise, Shift to rotate counter-clockwise. "
        "↓ for soft drop, Space for hard drop."
    )

    # User-facing game description
    game_description = (
        "A fast-paced falling block puzzle. Clear 10 lines in 60 seconds to advance through 3 "
        "increasingly difficult stages. Strategize your placements for big combo rewards!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Playfield dimensions
    PLAYFIELD_WIDTH = 10
    PLAYFIELD_HEIGHT = 20
    CELL_SIZE = 18

    # Colors
    COLOR_BG = (20, 20, 35)
    COLOR_GRID = (40, 40, 60)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (100, 200, 255)
    COLOR_GHOST = (255, 255, 255, 60)
    COLOR_WARN_BG = (100, 20, 20, 100)

    # Block shapes and colors (I, T, L, S)
    SHAPES = [
        ( # I
            [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 0), (2, 1), (2, 2), (2, 3)]],
            (0, 240, 240) # Cyan
        ),
        ( # T
            [[(0, 1), (1, 1), (2, 1), (1, 0)], [(1, 0), (1, 1), (1, 2), (2, 1)],
             [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (1, 1), (1, 2), (0, 1)]],
            (160, 0, 240) # Purple
        ),
        ( # L
            [[(0, 1), (1, 1), (2, 1), (2, 0)], [(1, 0), (1, 1), (1, 2), (2, 2)],
             [(0, 1), (1, 1), (2, 1), (0, 2)], [(1, 0), (1, 1), (1, 2), (0, 0)]],
            (240, 160, 0) # Orange
        ),
        ( # S
            [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
            (0, 240, 0) # Green
        )
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_title = pygame.font.SysFont("Consolas", 36, bold=True)
        
        # Playfield position
        self.playfield_x = (self.SCREEN_WIDTH - self.PLAYFIELD_WIDTH * self.CELL_SIZE) // 2
        self.playfield_y = (self.SCREEN_HEIGHT - self.PLAYFIELD_HEIGHT * self.CELL_SIZE) // 2

        # Initialize state variables
        self.playfield = None
        self.current_block = None
        self.next_block = None
        self.score = 0
        self.stage = 1
        self.total_lines_cleared = 0
        self.lines_cleared_this_stage = 0
        
        self.stage_timer = 0
        self.auto_drop_counter = 0
        self.base_auto_drop_speed = self.FPS
        self.current_auto_drop_speed = self.FPS

        self.game_over = False
        self.game_won = False
        self.steps = 0
        
        self.particles = []
        self.line_clear_flash = []
        
        self.space_was_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.playfield = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        self.score = 0
        self.stage = 1
        self.total_lines_cleared = 0
        self.game_over = False
        self.game_won = False
        self.steps = 0
        self.particles.clear()
        self.line_clear_flash.clear()
        self.space_was_held = False
        
        self._start_stage()
        self.next_block = self._new_block()
        self._spawn_block()

        return self._get_observation(), self._get_info()

    def _start_stage(self):
        self.stage_timer = 60 * self.FPS
        self.lines_cleared_this_stage = 0
        self.current_auto_drop_speed = self.base_auto_drop_speed - (self.stage - 1) * (0.05 * self.FPS) * self.PLAYFIELD_WIDTH
        self.auto_drop_counter = 0

    def step(self, action):
        reward = 0
        terminated = self.game_over or self.game_won

        if not terminated:
            self.steps += 1
            self.stage_timer -= 1
            
            # --- Handle Game Logic ---
            reward += self._handle_input(action)
            self._update_auto_drop()
            
            # --- Check for Termination ---
            if self.stage_timer <= 0:
                self.game_over = True
                terminated = True
            
            # --- Update Animations ---
            self._update_particles()
            if self.line_clear_flash:
                self.line_clear_flash[0] -= 1
                if self.line_clear_flash[0] <= 0:
                    self.line_clear_flash.clear()

        # Final check for termination state
        terminated = self.game_over or self.game_won
        
        # Calculate final rewards on termination
        if terminated and self.game_won:
            reward += 300 # Win game bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement and Rotation
        if movement == 1: # Rotate CW
            self._rotate_block(1)
        elif movement == 2: # Soft Drop
            self.auto_drop_counter += self.current_auto_drop_speed / 2 # Accelerate drop
        elif movement == 3: # Left
            self._move_block(-1, 0)
        elif movement == 4: # Right
            self._move_block(1, 0)
            
        if shift_pressed: # Rotate CCW
            self._rotate_block(-1)

        # Hard Drop (triggers on press, not hold)
        if space_pressed and not self.space_was_held:
            reward += self._hard_drop()
        self.space_was_held = space_pressed
        
        return reward

    def _update_auto_drop(self):
        self.auto_drop_counter += 1
        if self.auto_drop_counter >= self.current_auto_drop_speed:
            self.auto_drop_counter = 0
            if not self._move_block(0, 1):
                self._lock_block()

    def _new_block(self):
        shape_idx = self.np_random.integers(0, len(self.SHAPES))
        shape_data, color = self.SHAPES[shape_idx]
        return {
            "shape_idx": shape_idx,
            "rotation": 0,
            "pos": [self.PLAYFIELD_WIDTH // 2 - 2, 0],
            "shape": shape_data[0],
            "color": color
        }

    def _spawn_block(self):
        self.current_block = self.next_block
        self.next_block = self._new_block()
        if not self._is_valid_position(self.current_block["shape"], self.current_block["pos"]):
            self.game_over = True # Block spawned in an invalid position

    def _is_valid_position(self, shape, pos):
        for y_offset, x_offset in shape:
            x = pos[0] + x_offset
            y = pos[1] + y_offset
            if not (0 <= x < self.PLAYFIELD_WIDTH and 0 <= y < self.PLAYFIELD_HEIGHT):
                return False
            if self.playfield[y, x] != 0:
                return False
        return True

    def _move_block(self, dx, dy):
        new_pos = [self.current_block["pos"][0] + dx, self.current_block["pos"][1] + dy]
        if self._is_valid_position(self.current_block["shape"], new_pos):
            self.current_block["pos"] = new_pos
            return True
        return False

    def _rotate_block(self, direction):
        shape_data, _ = self.SHAPES[self.current_block["shape_idx"]]
        num_rotations = len(shape_data)
        new_rot = (self.current_block["rotation"] + direction) % num_rotations
        
        # Wall kick - try to shift position if rotation is blocked
        for dx in [0, 1, -1, 2, -2]:
            new_pos = [self.current_block["pos"][0] + dx, self.current_block["pos"][1]]
            if self._is_valid_position(shape_data[new_rot], new_pos):
                self.current_block["rotation"] = new_rot
                self.current_block["shape"] = shape_data[new_rot]
                self.current_block["pos"] = new_pos
                return

    def _hard_drop(self):
        ghost_y = self._get_ghost_position()
        self.current_block["pos"][1] = ghost_y
        return self._lock_block()

    def _lock_block(self):
        reward = 0.1 # Base reward for placing a block
        is_safe_placement = False
        for y_offset, x_offset in self.current_block["shape"]:
            x = self.current_block["pos"][0] + x_offset
            y = self.current_block["pos"][1] + y_offset
            if 0 <= x < self.PLAYFIELD_WIDTH and 0 <= y < self.PLAYFIELD_HEIGHT:
                self.playfield[y, x] = self.current_block["shape_idx"] + 1
                # Check for "safe" placement
                if y + 1 < self.PLAYFIELD_HEIGHT and self.playfield[y+1, x] != 0:
                    is_safe_placement = True

        if is_safe_placement:
            reward -= 0.02 # Penalty for safe placement
        
        # sound placeholder: # sfx_lock_block
        reward += self._clear_lines()
        self._spawn_block()
        self.auto_drop_counter = 0
        return reward

    def _clear_lines(self):
        lines_to_clear = []
        for r in range(self.PLAYFIELD_HEIGHT):
            if np.all(self.playfield[r, :] != 0):
                lines_to_clear.append(r)
        
        if not lines_to_clear:
            return 0

        # sound placeholder: # sfx_clear_line
        for r in lines_to_clear:
            self.playfield[r, :] = 0
            self._create_particles(r)
        
        self.line_clear_flash = [self.FPS // 4, lines_to_clear] # Flash for 1/4 second
        
        # Drop lines above
        lines_cleared_count = len(lines_to_clear)
        lines_to_clear.sort(reverse=True)
        for r in lines_to_clear:
            self.playfield[1:r+1, :] = self.playfield[0:r, :]
            self.playfield[0, :] = 0
            
        self.lines_cleared_this_stage += lines_cleared_count
        self.total_lines_cleared += lines_cleared_count
        self.score += [0, 1, 3, 7, 15][lines_cleared_count] * self.stage
        
        # Stage progression
        if self.lines_cleared_this_stage >= 10:
            self.stage += 1
            if self.stage > 3:
                self.game_won = True
                return 0 # Win bonus added in step()
            else:
                self._start_stage()
                # sound placeholder: # sfx_stage_clear
                return 100 # Stage clear bonus

        return [0, 1, 3, 7, 15][lines_cleared_count]

    def _get_ghost_position(self):
        y = self.current_block["pos"][1]
        while self._is_valid_position(self.current_block["shape"], [self.current_block["pos"][0], y + 1]):
            y += 1
        return y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage, "lines": self.total_lines_cleared}

    def _render_game(self):
        # Draw playfield border and grid
        pf_rect = pygame.Rect(self.playfield_x, self.playfield_y,
                              self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.PLAYFIELD_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, pf_rect, 2)
        for i in range(1, self.PLAYFIELD_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.playfield_x + i * self.CELL_SIZE, self.playfield_y),
                             (self.playfield_x + i * self.CELL_SIZE, self.playfield_y + self.PLAYFIELD_HEIGHT * self.CELL_SIZE))
        for i in range(1, self.PLAYFIELD_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.playfield_x, self.playfield_y + i * self.CELL_SIZE),
                             (self.playfield_x + self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.playfield_y + i * self.CELL_SIZE))
        
        # Draw warning zone
        warn_alpha = int(100 * (0.5 + 0.5 * math.sin(self.steps * 0.2))) # Pulsing effect
        warn_surf = pygame.Surface((pf_rect.width, 4 * self.CELL_SIZE), pygame.SRCALPHA)
        warn_surf.fill((100, 20, 20, warn_alpha))
        self.screen.blit(warn_surf, (pf_rect.x, pf_rect.y))
        
        # Draw locked blocks
        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                if self.playfield[r, c] != 0:
                    color = self.SHAPES[int(self.playfield[r, c] - 1)][1]
                    self._draw_block_cell(c, r, color)
        
        # Draw line clear flash
        if self.line_clear_flash:
            _, lines = self.line_clear_flash
            flash_surf = pygame.Surface((pf_rect.width, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surf.fill((255, 255, 255, 180))
            for r in lines:
                self.screen.blit(flash_surf, (pf_rect.x, self.playfield_y + r * self.CELL_SIZE))

        # Draw ghost piece
        if self.current_block and not self.game_over:
            ghost_y = self._get_ghost_position()
            ghost_pos = [self.current_block["pos"][0], ghost_y]
            for y_offset, x_offset in self.current_block["shape"]:
                x = ghost_pos[0] + x_offset
                y = ghost_pos[1] + y_offset
                rect = pygame.Rect(self.playfield_x + x * self.CELL_SIZE,
                                   self.playfield_y + y * self.CELL_SIZE,
                                   self.CELL_SIZE, self.CELL_SIZE)
                pygame.gfxdraw.box(self.screen, rect, self.COLOR_GHOST)

        # Draw current block
        if self.current_block and not self.game_over:
            for y_offset, x_offset in self.current_block["shape"]:
                self._draw_block_cell(self.current_block["pos"][0] + x_offset,
                                      self.current_block["pos"][1] + y_offset,
                                      self.current_block["color"])
        
        self._render_particles()

    def _draw_block_cell(self, c, r, color):
        rect = pygame.Rect(self.playfield_x + c * self.CELL_SIZE + 1,
                           self.playfield_y + r * self.CELL_SIZE + 1,
                           self.CELL_SIZE - 2, self.CELL_SIZE - 2)
        pygame.draw.rect(self.screen, color, rect)
        darker_color = tuple(max(0, val - 50) for val in color)
        pygame.draw.rect(self.screen, darker_color, rect, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Stage and Lines
        stage_text = self.font_main.render(f"STAGE: {self.stage}/3", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (20, 50))
        lines_text = self.font_small.render(f"LINES: {self.lines_cleared_this_stage}/10", True, self.COLOR_UI_ACCENT)
        self.screen.blit(lines_text, (20, 80))

        # Timer
        time_left = max(0, self.stage_timer / self.FPS)
        timer_color = self.COLOR_UI_TEXT if time_left > 10 else (255, 80, 80)
        timer_text = self.font_main.render(f"TIME: {time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 20, 20))

        # Next block preview
        next_text = self.font_small.render("NEXT:", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 120, self.SCREEN_HEIGHT - 120))
        if self.next_block:
            shape, color = self.next_block["shape"], self.next_block["color"]
            for y_offset, x_offset in shape:
                rect = pygame.Rect(self.SCREEN_WIDTH - 110 + x_offset * self.CELL_SIZE,
                                   self.SCREEN_HEIGHT - 90 + y_offset * self.CELL_SIZE,
                                   self.CELL_SIZE -2, self.CELL_SIZE - 2)
                pygame.draw.rect(self.screen, color, rect)
                darker_color = tuple(max(0, val - 50) for val in color)
                pygame.draw.rect(self.screen, darker_color, rect, 2)
        
        # Game Over / Win message
        if self.game_over or self.game_won:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            msg_render = self.font_title.render(msg, True, self.COLOR_UI_ACCENT)
            self.screen.blit(msg_render, (self.SCREEN_WIDTH//2 - msg_render.get_width()//2, self.SCREEN_HEIGHT//2 - 50))
            score_render = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            self.screen.blit(score_render, (self.SCREEN_WIDTH//2 - score_render.get_width()//2, self.SCREEN_HEIGHT//2))
            
    def _create_particles(self, row):
        for _ in range(40):
            px = self.playfield_x + self.np_random.uniform(0, self.PLAYFIELD_WIDTH * self.CELL_SIZE)
            py = self.playfield_y + row * self.CELL_SIZE + self.np_random.uniform(0, self.CELL_SIZE)
            vel_angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(1, 4)
            vx = math.cos(vel_angle) * vel_mag
            vy = math.sin(vel_angle) * vel_mag
            life = self.np_random.integers(self.FPS // 2, self.FPS)
            size = self.np_random.uniform(2, 5)
            color = random.choice([(255, 255, 220), (200, 200, 255), (255, 220, 200)])
            self.particles.append({"pos": [px, py], "vel": [vx, vy], "life": life, "size": size, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            p["size"] = max(0, p["size"] - 0.1)
        self.particles = [p for p in self.particles if p["life"] > 0 and p["size"] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / self.FPS))
            if alpha > 0:
                pygame.gfxdraw.box(
                    self.screen,
                    pygame.Rect(int(p["pos"][0]), int(p["pos"][1]), int(p["size"]), int(p["size"])),
                    p["color"] + (alpha,)
                )

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the rendered frames
    pygame.display.set_caption("Falling Block Puzzle")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Map keyboard inputs to the MultiDiscrete action space
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # The env will now just return the final state. 
            # We can wait for a reset key press.
        
        # Display the frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    pygame.quit()