
# Generated: 2025-08-28T03:08:12.984841
# Source Brief: brief_04822.md
# Brief Index: 4822

        
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
        "Controls: Arrow keys to move cursor. Space to select/swap. Shift to deselect."
    )

    game_description = (
        "Match cascading gems in a fast-paced isometric puzzle game to reach a target score before running out of moves."
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    TOTAL_MOVES = 20
    WIN_SCORE = 1000
    MAX_STEPS = 1000 # Safety break

    # Visuals
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_W = 48
    TILE_H = 24
    GEM_SIZE = 18

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_SELECT = (255, 255, 255, 200)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 150, 80),  # Orange
    ]

    ANIMATION_SPEED = 4.0 # higher is faster

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 30)
        self.font_msg = pygame.font.Font(None, 48)
        
        self.grid_offset_x = self.SCREEN_WIDTH // 2
        self.grid_offset_y = 80

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.TOTAL_MOVES
        self.game_over = False
        
        self._initialize_board()
        
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem = None
        
        self.game_state = "IDLE" # IDLE, SWAP, POP, FALL, SHUFFLE
        self.animation_timer = 0.0
        self.animation_data = {}
        self.particles = []
        self.turn_reward = 0.0

        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        if self.game_state != "IDLE":
            reward, terminated = self._update_animation()
        else:
            self._handle_player_action(movement, space_press, shift_press)
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_action(self, movement, space_press, shift_press):
        # Handle cursor movement
        if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        if movement == 2 and self.cursor_pos[0] < self.GRID_HEIGHT - 1: self.cursor_pos[0] += 1
        if movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        if movement == 4 and self.cursor_pos[1] < self.GRID_WIDTH - 1: self.cursor_pos[1] += 1

        if shift_press and self.selected_gem:
            self.selected_gem = None
            # sfx: deselect

        if space_press:
            if not self.selected_gem:
                self.selected_gem = list(self.cursor_pos)
                # sfx: select
            else:
                # Attempt to swap
                r1, c1 = self.selected_gem
                r2, c2 = self.cursor_pos
                
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    # Valid adjacent swap attempt
                    self.moves_left -= 1
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    self.animation_data = {'pos1': [r1, c1], 'pos2': [r2, c2]}
                    self.game_state = "SWAP"
                    self.animation_timer = 0.0
                    self.selected_gem = None
                    # sfx: swap
                else:
                    # Invalid swap, just re-select
                    self.selected_gem = list(self.cursor_pos)
                    # sfx: error

    def _update_animation(self):
        self.animation_timer += self.ANIMATION_SPEED / 30.0 # Assuming 30fps
        reward, terminated = 0, False

        if self.animation_timer >= 1.0:
            self.animation_timer = 0.0
            
            if self.game_state == "SWAP":
                matches = self._find_all_matches()
                if matches:
                    self.animation_data['matches'] = matches
                    self.game_state = "POP"
                else:
                    # Invalid move, swap back
                    r1, c1 = self.animation_data['pos1']
                    r2, c2 = self.animation_data['pos2']
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    self.game_state = "IDLE"
                    self.turn_reward = -0.1
                    reward = self.turn_reward
                    self.turn_reward = 0
            
            elif self.game_state == "POP":
                matches = self.animation_data['matches']
                
                # Calculate reward for this pop
                num_cleared = len(matches)
                self.turn_reward += num_cleared
                if num_cleared == 4: self.turn_reward += 5
                if num_cleared >= 5: self.turn_reward += 10
                self.score += num_cleared * 10 # Simple score for display
                
                # Create particles and remove gems
                for r, c in matches:
                    self._create_particles(r, c)
                    self.grid[r, c] = -1 # Mark for removal
                # sfx: match_pop
                
                self.animation_data = {'fall_map': self._apply_gravity()}
                self.game_state = "FALL"

            elif self.game_state == "FALL":
                self._refill_top_rows()
                matches = self._find_all_matches()
                if matches:
                    self.animation_data['matches'] = matches
                    self.game_state = "POP" # Chain reaction
                else:
                    if not self._check_possible_moves():
                        self._shuffle_board()
                        self.game_state = "SHUFFLE"
                    else:
                        self.game_state = "IDLE"

            elif self.game_state == "SHUFFLE":
                self.game_state = "IDLE"

            # If the turn has ended (transitioned to IDLE or SHUFFLE)
            if self.game_state in ["IDLE", "SHUFFLE"]:
                reward = self.turn_reward
                self.turn_reward = 0
                if self.score >= self.WIN_SCORE or self.moves_left <= 0:
                    terminated = True
                    reward += 100 if self.score >= self.WIN_SCORE else -100
        
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid_background()
        self._draw_gems()
        self._draw_cursor_and_selection()
        self._update_and_draw_particles()
        self._draw_ui()
        if self.game_over:
            self._draw_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        moves_made = self.TOTAL_MOVES - self.moves_left
        return {"score": self.score, "steps": moves_made}

    # --- Game Logic Helpers ---

    def _initialize_board(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_all_matches()
            if not matches:
                if self._check_possible_moves():
                    break
                else:
                    self._shuffle_board(no_anim=True)
            else:
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type == -1: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c+1] == gem_type and self.grid[r, c+2] == gem_type:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r+1, c] == gem_type and self.grid[r+2, c] == gem_type:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _check_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_all_matches():
                        self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                        return True
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_all_matches():
                        self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                        return True
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
        return False

    def _apply_gravity(self):
        fall_map = {} # { (from_r, from_c): (to_r, to_c) }
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        fall_map[(r, c)] = (empty_row, c)
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
        return fall_map
    
    def _refill_top_rows(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _shuffle_board(self, no_anim=False):
        gem_list = self.grid.flatten().tolist()
        self.np_random.shuffle(gem_list)
        self.grid = np.array(gem_list).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        if no_anim: return

        while True:
            matches = self._find_all_matches()
            if not matches:
                if self._check_possible_moves():
                    break
                else: # Reshuffle again if still no moves
                    self.np_random.shuffle(gem_list)
                    self.grid = np.array(gem_list).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
            else: # Remove matches from shuffle
                for r_m, c_m in matches: self.grid[r_m, c_m] = self.np_random.integers(0, self.NUM_GEM_TYPES)
        # sfx: shuffle

    # --- Rendering Helpers ---

    def _iso_to_screen(self, r, c):
        x = self.grid_offset_x + (c - r) * self.TILE_W / 2
        y = self.grid_offset_y + (c + r) * self.TILE_H / 2
        return int(x), int(y)

    def _draw_grid_background(self):
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.GRID_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.GRID_HEIGHT, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

    def _draw_gem(self, r, c, gem_type, pos_override=None):
        if gem_type < 0: return
        
        x, y = pos_override if pos_override else self._iso_to_screen(r, c)
        color = self.GEM_COLORS[gem_type]
        
        # Simple shapes for different gems
        if gem_type == 0: # Circle
            pygame.gfxdraw.aacircle(self.screen, x, y, self.GEM_SIZE // 2, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.GEM_SIZE // 2, color)
        elif gem_type == 1: # Square
            rect = pygame.Rect(x - self.GEM_SIZE//2, y - self.GEM_SIZE//2, self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        elif gem_type == 2: # Diamond
            points = [(x, y - self.GEM_SIZE//2), (x + self.GEM_SIZE//2, y), (x, y + self.GEM_SIZE//2), (x - self.GEM_SIZE//2, y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 3: # Triangle Up
            points = [(x, y - self.GEM_SIZE//2), (x + self.GEM_SIZE//2, y + self.GEM_SIZE//2), (x - self.GEM_SIZE//2, y + self.GEM_SIZE//2)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4: # Hexagon
            radius = self.GEM_SIZE / 2
            points = [(x + radius * math.cos(math.pi/3 * i), y + radius * math.sin(math.pi/3 * i)) for i in range(6)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else: # Triangle Down
            points = [(x, y + self.GEM_SIZE//2), (x + self.GEM_SIZE//2, y - self.GEM_SIZE//2), (x - self.GEM_SIZE//2, y - self.GEM_SIZE//2)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_gems(self):
        drawn_gems = set()
        t = self.animation_timer

        # Draw animated gems first
        if self.game_state == "SWAP":
            r1, c1 = self.animation_data['pos1']
            r2, c2 = self.animation_data['pos2']
            g1, g2 = self.grid[r1, c1], self.grid[r2, c2]
            
            x1, y1 = self._iso_to_screen(r1, c1)
            x2, y2 = self._iso_to_screen(r2, c2)
            
            # Interpolate positions
            ix1 = int(x1 + (x2 - x1) * t)
            iy1 = int(y1 + (y2 - y1) * t)
            ix2 = int(x2 + (x1 - x2) * t)
            iy2 = int(y2 + (y1 - y2) * t)
            
            self._draw_gem(r1, c1, g1, (ix2, iy2))
            self._draw_gem(r2, c2, g2, (ix1, iy1))
            drawn_gems.add((r1, c1))
            drawn_gems.add((r2, c2))

        elif self.game_state == "FALL":
            fall_map = self.animation_data['fall_map']
            for (r1, c1), (r2, c2) in fall_map.items():
                x1, y1 = self._iso_to_screen(r1, c1)
                x2, y2 = self._iso_to_screen(r2, c2)
                ix = int(x1 + (x2 - x1) * t)
                iy = int(y1 + (y2 - y1) * t)
                self._draw_gem(r2, c2, self.grid[r2, c2], (ix, iy))
                drawn_gems.add((r2, c2))

        # Draw static gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r, c) not in drawn_gems:
                    gem_type = self.grid[r,c]
                    if self.game_state == "POP" and 'matches' in self.animation_data and (r,c) in self.animation_data['matches']:
                        # Shrink popping gems
                        scale = 1.0 - t
                        size_backup = self.GEM_SIZE
                        self.GEM_SIZE = int(size_backup * scale)
                        self._draw_gem(r, c, gem_type)
                        self.GEM_SIZE = size_backup
                    else:
                        self._draw_gem(r, c, gem_type)
    
    def _draw_cursor_and_selection(self):
        # Draw selection highlight
        if self.selected_gem:
            r, c = self.selected_gem
            x, y = self._iso_to_screen(r, c)
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            radius = int(self.GEM_SIZE * 0.8 + pulse * 4)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_SELECT)
        
        # Draw cursor
        r, c = self.cursor_pos
        x, y = self._iso_to_screen(r, c)
        points = [(x, y - self.TILE_H//2), (x + self.TILE_W//2, y), (x, y + self.TILE_H//2), (x - self.TILE_W//2, y)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)

    def _create_particles(self, r, c):
        x, y = self._iso_to_screen(r, c)
        gem_type = self.grid[r, c]
        if gem_type < 0: return
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([[x, y], vel, color, lifetime])

    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[1][1] += 0.2     # gravity
            p[3] -= 1          # lifetime
            size = max(0, int(p[3] / 5.0))
            if size > 0:
                pygame.draw.circle(self.screen, p[2], p[0], size)
        self.particles = [p for p in self.particles if p[3] > 0]

    def _draw_ui(self):
        score_surf = self.font_ui.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (10, 10))
        
        moves_surf = self.font_ui.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH - moves_surf.get_width() - 10, 10))

    def _draw_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        
        msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
        text_surf = self.font_msg.render(msg, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        s.blit(text_surf, text_rect)
        self.screen.blit(s, (0,0))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use this to test the game with keyboard controls
    
    running = True
    game_over = False
    total_reward = 0
    
    # Map Pygame keys to MultiDiscrete actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Setup a window to display the rendered frames
    pygame.display.set_caption("Gemstone Cascade")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        # Get keyboard input
        movement = 0
        space_held = False
        shift_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                game_over = False
                total_reward = 0

        if not game_over:
            keys = pygame.key.get_pressed()
            for key, move_action in key_to_action.items():
                if keys[key]:
                    movement = move_action
                    break # Prioritize one movement key
            
            if keys[pygame.K_SPACE]:
                space_held = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = True
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated and not game_over:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            game_over = True
            
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(30)
        
    env.close()