
# Generated: 2025-08-27T14:51:16.701685
# Source Brief: brief_00808.md
# Brief Index: 808

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←/→ to move, ↑/↓ to rotate. Hold Shift for soft drop, press Space for hard drop."
    )

    game_description = (
        "A fast-paced, procedurally generated falling block puzzle where strategic risk-taking leads to higher scores."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 20
        self.BLOCK_SIZE = 18
        
        # Play area position
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 3
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255)
        self.TETROMINO_COLORS = [
            (0, 0, 0),          # 0: Empty
            (0, 240, 240),      # 1: I (Cyan)
            (240, 240, 0),      # 2: O (Yellow)
            (160, 0, 240),      # 3: T (Purple)
            (0, 0, 240),        # 4: J (Blue)
            (240, 160, 0),      # 5: L (Orange)
            (0, 240, 0),        # 6: S (Green)
            (240, 0, 0),        # 7: Z (Red)
        ]

        # Tetromino Shapes [shape][rotation]
        self.SHAPES = [
            [], # Empty
            [[(0,1),(1,1),(2,1),(3,1)], [(2,0),(2,1),(2,2),(2,3)]], # I
            [[(1,1),(2,1),(1,2),(2,2)]], # O
            [[(1,0),(0,1),(1,1),(2,1)], [(1,0),(1,1),(2,1),(1,2)], [(0,1),(1,1),(2,1),(1,2)], [(1,0),(0,1),(1,1),(1,2)]], # T
            [[(0,0),(0,1),(1,1),(2,1)], [(1,0),(2,0),(1,1),(1,2)], [(0,1),(1,1),(2,1),(2,2)], [(1,0),(1,1),(0,2),(1,2)]], # J
            [[(2,0),(0,1),(1,1),(2,1)], [(1,0),(1,1),(1,2),(2,2)], [(0,1),(1,1),(2,1),(0,2)], [(0,0),(1,0),(1,1),(1,2)]], # L
            [[(1,0),(2,0),(0,1),(1,1)], [(0,0),(0,1),(1,1),(1,2)]], # S
            [[(0,0),(1,0),(1,1),(2,1)], [(2,0),(1,1),(2,1),(1,2)]]  # Z
        ]

        # Game State
        self.grid = None
        self.current_piece = None
        self.next_piece_shape = None
        self.fall_progress = 0.0
        self.fall_speed = 0.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.line_clear_animation = []
        
        # Action handling timers for better game feel
        self.move_cooldown = 0
        self.MOVE_DELAY = 4 # frames

        self.rng = None
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0.02 # Slower start
        self.fall_progress = 0.0
        self.particles = []
        self.line_clear_animation = []
        self.move_cooldown = 0

        self.next_piece_shape = self.rng.integers(1, len(self.SHAPES))
        self._spawn_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for time passing

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Action Handling ---
        hard_dropped = False
        if space_held:
            reward_change, score_change, _ = self._hard_drop()
            reward += reward_change
            self.score += score_change
            hard_dropped = True
        else:
            # Handle horizontal movement with cooldown
            if self.move_cooldown > 0:
                self.move_cooldown -= 1
            
            if self.move_cooldown == 0:
                if movement == 3: # Left
                    self._move(-1)
                    self.move_cooldown = self.MOVE_DELAY
                elif movement == 4: # Right
                    self._move(1)
                    self.move_cooldown = self.MOVE_DELAY

            # Handle rotation
            if movement == 1: # Rotate CW
                self._rotate(1)
            elif movement == 2: # Rotate CCW
                self._rotate(-1)
        
        # --- Game Logic Update ---
        if not hard_dropped:
            current_fall_speed = self.fall_speed * 5 if shift_held else self.fall_speed
            self.fall_progress += current_fall_speed
            
            if self.fall_progress >= 1.0:
                self.fall_progress -= 1.0
                
                # Move piece down
                new_y = self.current_piece['y'] + 1
                if self._is_valid_position(self.current_piece['shape_idx'], self.current_piece['rotation'], self.current_piece['x'], new_y):
                    self.current_piece['y'] = new_y
                else:
                    # Place piece and spawn next
                    reward_change, score_change, overhang_penalty = self._place_piece()
                    reward += reward_change + overhang_penalty
                    self.score += score_change
        
        # --- Update Difficulty ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.fall_speed = min(1.0, self.fall_speed + 0.01)

        terminated = self._check_termination()
        if terminated and not self.game_over:
             # Win condition
             reward += 100
        elif self.game_over:
             # Lose condition
             reward += -50
        
        if self.steps >= 10000:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Game Logic Helpers ---
    def _spawn_piece(self):
        self.current_piece = {
            'shape_idx': self.next_piece_shape,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0
        }
        self.next_piece_shape = self.rng.integers(1, len(self.SHAPES))
        
        # Game over check
        if not self._is_valid_position(self.current_piece['shape_idx'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y']):
            self.game_over = True

    def _get_piece_coords(self, shape_idx, rotation, x, y):
        shape = self.SHAPES[shape_idx][rotation % len(self.SHAPES[shape_idx])]
        return [(c + x, r + y) for c, r in shape]

    def _is_valid_position(self, shape_idx, rotation, x, y):
        coords = self._get_piece_coords(shape_idx, rotation, x, y)
        for c, r in coords:
            if not (0 <= c < self.GRID_WIDTH and 0 <= r < self.GRID_HEIGHT):
                return False # Out of bounds
            if self.grid[r, c] != 0:
                return False # Collision with existing block
        return True

    def _move(self, dx):
        if self.current_piece and self._is_valid_position(self.current_piece['shape_idx'], self.current_piece['rotation'], self.current_piece['x'] + dx, self.current_piece['y']):
            self.current_piece['x'] += dx

    def _rotate(self, dr):
        if not self.current_piece: return
        
        new_rot = (self.current_piece['rotation'] + dr) % len(self.SHAPES[self.current_piece['shape_idx']])
        
        # Try simple rotation
        if self._is_valid_position(self.current_piece['shape_idx'], new_rot, self.current_piece['x'], self.current_piece['y']):
            self.current_piece['rotation'] = new_rot
            return

        # Try wall kicks
        for kick_x in [-1, 1, -2, 2]:
            if self._is_valid_position(self.current_piece['shape_idx'], new_rot, self.current_piece['x'] + kick_x, self.current_piece['y']):
                self.current_piece['x'] += kick_x
                self.current_piece['rotation'] = new_rot
                return

    def _hard_drop(self):
        if not self.current_piece: return 0, 0, 0
        
        # Find landing spot
        y = self.current_piece['y']
        while self._is_valid_position(self.current_piece['shape_idx'], self.current_piece['rotation'], self.current_piece['x'], y + 1):
            y += 1
        self.current_piece['y'] = y
        
        # Place piece and get rewards
        return self._place_piece()

    def _place_piece(self):
        if not self.current_piece: return 0, 0, 0
        
        coords = self._get_piece_coords(self.current_piece['shape_idx'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y'])
        
        # Check for overhangs (for reward calculation)
        overhang_penalty = 0
        for c, r in coords:
            if r + 1 < self.GRID_HEIGHT and self.grid[r + 1, c] == 0:
                 overhang_penalty -= 0.2
        
        # Place piece on grid
        for c, r in coords:
            if 0 <= c < self.GRID_WIDTH and 0 <= r < self.GRID_HEIGHT:
                self.grid[r, c] = self.current_piece['shape_idx']
        
        # Create landing particles
        self._create_particles(coords, self.TETROMINO_COLORS[self.current_piece['shape_idx']], 10)
        
        self.current_piece = None
        reward, score = self._clear_lines()
        self._spawn_piece()
        
        return reward, score, overhang_penalty

    def _clear_lines(self):
        lines_cleared = 0
        full_rows = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                lines_cleared += 1
                full_rows.append(r)
        
        if lines_cleared > 0:
            for r in full_rows:
                # Add to animation
                self.line_clear_animation.append({'y': r, 'timer': 10})
                # Create particles
                coords = [(c,r) for c in range(self.GRID_WIDTH)]
                self._create_particles(coords, self.COLOR_FLASH, 20)

            # Shift grid down
            self.grid = np.delete(self.grid, full_rows, axis=0)
            new_rows = np.zeros((lines_cleared, self.GRID_WIDTH), dtype=int)
            self.grid = np.vstack((new_rows, self.grid))

        # Calculate reward and score
        reward_map = {0: 0, 1: 1, 2: 4, 3: 7, 4: 12} # Base + Bonus
        score_map = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        return reward_map.get(lines_cleared, 0), score_map.get(lines_cleared, 0)

    def _check_termination(self):
        return self.game_over or self.score >= 1000

    # --- Rendering Helpers ---
    def _draw_block(self, surface, x, y, color):
        """Draws a single block with a 3D effect."""
        r, g, b = color
        darker_color = (max(0, r - 70), max(0, g - 70), max(0, b - 70))
        lighter_color = (min(255, r + 70), min(255, g + 70), min(255, b + 70))
        
        # Main rect
        pygame.draw.rect(surface, darker_color, (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE))
        # Inner rect
        pygame.draw.rect(surface, color, (x + 1, y + 1, self.BLOCK_SIZE - 2, self.BLOCK_SIZE - 2))
        # Highlight
        pygame.draw.line(surface, lighter_color, (x, y), (x + self.BLOCK_SIZE - 1, y), 1)
        pygame.draw.line(surface, lighter_color, (x, y), (x, y + self.BLOCK_SIZE - 1), 1)


    def _render_game(self):
        # Draw grid background
        grid_surface = pygame.Surface((self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE))
        grid_surface.set_alpha(150)
        grid_surface.fill(self.COLOR_GRID)
        for x in range(0, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE):
            pygame.draw.line(grid_surface, (50,50,70), (x, 0), (x, self.GRID_HEIGHT * self.BLOCK_SIZE))
        for y in range(0, self.GRID_HEIGHT * self.BLOCK_SIZE, self.BLOCK_SIZE):
            pygame.draw.line(grid_surface, (50,50,70), (0, y), (self.GRID_WIDTH * self.BLOCK_SIZE, y))
        self.screen.blit(grid_surface, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET))
        
        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color = self.TETROMINO_COLORS[self.grid[r, c]]
                    self._draw_block(self.screen, self.GRID_X_OFFSET + c * self.BLOCK_SIZE, self.GRID_Y_OFFSET + r * self.BLOCK_SIZE, color)
        
        # Draw line clear animation
        active_anims = []
        for anim in self.line_clear_animation:
            y = self.GRID_Y_OFFSET + anim['y'] * self.BLOCK_SIZE
            alpha = int(255 * (anim['timer'] / 10))
            flash_surface = pygame.Surface((self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE))
            flash_surface.set_alpha(alpha)
            flash_surface.fill(self.COLOR_FLASH)
            self.screen.blit(flash_surface, (self.GRID_X_OFFSET, y))
            anim['timer'] -= 1
            if anim['timer'] > 0:
                active_anims.append(anim)
        self.line_clear_animation = active_anims

        # Draw current piece and ghost piece
        if self.current_piece and not self.game_over:
            color = self.TETROMINO_COLORS[self.current_piece['shape_idx']]
            
            # Ghost piece
            ghost_y = self.current_piece['y']
            while self._is_valid_position(self.current_piece['shape_idx'], self.current_piece['rotation'], self.current_piece['x'], ghost_y + 1):
                ghost_y += 1
            ghost_coords = self._get_piece_coords(self.current_piece['shape_idx'], self.current_piece['rotation'], self.current_piece['x'], ghost_y)
            for c, r in ghost_coords:
                ghost_color = (color[0], color[1], color[2], 80)
                s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
                s.fill(ghost_color)
                self.screen.blit(s, (self.GRID_X_OFFSET + c * self.BLOCK_SIZE, self.GRID_Y_OFFSET + r * self.BLOCK_SIZE))

            # Actual piece
            piece_coords = self._get_piece_coords(self.current_piece['shape_idx'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y'])
            for c, r in piece_coords:
                self._draw_block(self.screen, self.GRID_X_OFFSET + c * self.BLOCK_SIZE, self.GRID_Y_OFFSET + r * self.BLOCK_SIZE, color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Next Piece
        next_box_x = self.GRID_X_OFFSET + self.GRID_WIDTH * self.BLOCK_SIZE + 40
        next_box_y = self.GRID_Y_OFFSET
        
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (next_box_x, next_box_y))
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (next_box_x - 5, next_box_y + 25, 4 * self.BLOCK_SIZE + 10, 4 * self.BLOCK_SIZE + 10), 0, 5)
        
        if self.next_piece_shape:
            shape_coords = self.SHAPES[self.next_piece_shape][0]
            color = self.TETROMINO_COLORS[self.next_piece_shape]
            for c, r in shape_coords:
                # Center the piece in the preview box
                draw_x = next_box_x + c * self.BLOCK_SIZE
                draw_y = next_box_y + 40 + r * self.BLOCK_SIZE
                self._draw_block(self.screen, draw_x, draw_y, color)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            end_text_str = "YOU WON!" if self.score >= 1000 else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_FLASH)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, coords, color, count_per_block):
        for c, r in coords:
            for _ in range(count_per_block):
                x = self.GRID_X_OFFSET + (c + 0.5) * self.BLOCK_SIZE
                y = self.GRID_Y_OFFSET + (r + 0.5) * self.BLOCK_SIZE
                angle = self.rng.uniform(0, 2 * math.pi)
                speed = self.rng.uniform(1, 3)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                lifetime = self.rng.integers(15, 30)
                self.particles.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] > 0:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                s = pygame.Surface((3, 3), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (1, 1), 1)
                self.screen.blit(s, (int(p['x']), int(p['y'])))
                active_particles.append(p)
        self.particles = active_particles

    def validate_implementation(self):
        '''Call this at the end of __init__ to verify implementation:'''
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Gymnasium Block Puzzle")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_SPACE:
                    space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
            
            # Key releases
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                    movement = 0
                elif event.key == pygame.K_SPACE:
                    space_held = 0
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift_held = 0
        
        # Construct action and step environment
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # After a hard drop, reset the space key to prevent it from being held down
        if space_held == 1:
            space_held = 0

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS

    pygame.quit()