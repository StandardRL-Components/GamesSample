
# Generated: 2025-08-27T12:50:24.768744
# Source Brief: brief_00170.md
# Brief Index: 170

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then move to an adjacent gem and press Space again to swap. Hold Shift to reshuffle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems in a grid to create matches of 3 or more, aiming to reach a target score before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    NUM_GEM_TYPES = 6
    MIN_MATCH_LENGTH = 3
    TARGET_SCORE = 1000
    MAX_MOVES = 50
    MAX_STEPS = 500  # An action is a full turn, so this is 500 turns.

    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_AREA_HEIGHT = 380
    UI_AREA_HEIGHT = 20

    COLOR_BG = (20, 25, 40)
    COLOR_GRID_BG = (30, 35, 50)
    COLOR_GRID_LINES = (50, 55, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 160, 80),  # Orange
    ]

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
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.cell_size = self.GRID_AREA_HEIGHT // self.GRID_HEIGHT
        self.grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.cell_size) // 2
        self.grid_offset_y = self.UI_AREA_HEIGHT
        self.gem_radius = int(self.cell_size * 0.4)

        self.grid = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selected_gem = None
        self.particles = []
        self.swap_animation = None
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._create_and_validate_grid()
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem = None
        self.particles = []
        self.swap_animation = None
        self.prev_space_held = 0
        self.prev_shift_held = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self.steps += 1
        reward = 0
        self.swap_animation = None

        # Update particle states from previous turn
        self._update_particles()

        # Handle Input and resolve the entire turn in one go
        reward += self._handle_input(movement, space_pressed, shift_pressed)

        # Update previous button states for next step
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Check for termination conditions
        terminated = (self.score >= self.TARGET_SCORE) or (self.moves_left <= 0) or (self.steps >= self.MAX_STEPS)
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.TARGET_SCORE:
                reward += 100  # Win bonus
            else:
                reward -= 50  # Loss penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Cursor Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # Shift to Reshuffle
        if shift_pressed and self.moves_left > 0:
            self.moves_left -= 1
            # Sound: reshuffle.wav
            self._create_and_validate_grid()
            return 0

        # Space to Select/Swap
        if space_pressed:
            cx, cy = self.cursor_pos
            if self.selected_gem is None:
                self.selected_gem = (cx, cy)
                # Sound: select_gem.wav
            else:
                sx, sy = self.selected_gem
                if (cx, cy) == (sx, sy):
                    self.selected_gem = None
                elif abs(cx - sx) + abs(cy - sy) == 1:
                    return self._attempt_swap((sx, sy), (cx, cy))
                else:
                    self.selected_gem = (cx, cy)
                    # Sound: select_gem.wav
        return 0

    def _attempt_swap(self, pos1, pos2):
        self.moves_left -= 1
        self.selected_gem = None
        
        # Perform swap
        y1, x1 = pos1[1], pos1[0]
        y2, x2 = pos2[1], pos2[0]
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
        
        # Set up animation data for rendering
        self.swap_animation = {"pos1": pos1, "pos2": pos2, "is_valid": False}

        matches = self._find_all_matches()
        if matches:
            # Sound: match_found.wav
            self.swap_animation["is_valid"] = True
            return self._process_chain_reactions()
        else:
            # Invalid swap, swap back
            # Sound: invalid_swap.wav
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            return -0.1

    def _process_chain_reactions(self):
        total_reward = 0
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            reward, num_cleared = self._clear_matches(matches)
            total_reward += reward
            self.score += num_cleared
            
            self._apply_gravity()
            self._refill_grid()

        if not self._find_possible_moves():
            # Sound: board_reshuffle_auto.wav
            self._create_and_validate_grid()

        return total_reward
    
    def _create_and_validate_grid(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_all_matches() and self._find_possible_moves():
                break

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_spaces = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] == 0:
                    empty_spaces += 1
                elif empty_spaces > 0:
                    self.grid[y + empty_spaces, x] = self.grid[y, x]
                    self.grid[y, x] = 0

    def _refill_grid(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[y, x] == 0:
                    self.grid[y, x] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _find_all_matches(self, grid=None):
        if grid is None: grid = self.grid
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return list(matches)

    def _find_possible_moves(self, grid=None):
        if grid is None: grid = self.grid
        temp_grid = grid.copy()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if c < self.GRID_WIDTH - 1:
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_all_matches(grid=temp_grid): return True
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                if r < self.GRID_HEIGHT - 1:
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_all_matches(grid=temp_grid): return True
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
        return False

    def _clear_matches(self, matches):
        num_cleared = len(set(matches))
        reward = num_cleared
        if num_cleared == 4: reward += 10
        elif num_cleared >= 5: reward += 25
        
        for c, r in set(matches):
            gem_type = self.grid[r, c]
            if gem_type > 0:
                color = self.GEM_COLORS[gem_type - 1]
                for _ in range(15):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(1, 4)
                    self.particles.append({
                        "pos": [self.grid_offset_x + c * self.cell_size + self.cell_size / 2, self.grid_offset_y + r * self.cell_size + self.cell_size / 2],
                        "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                        "life": random.randint(15, 30), "color": color
                    })
            self.grid[r, c] = 0
        return reward, num_cleared

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1
            p["life"] -= 1
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_game(self):
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.GRID_WIDTH * self.cell_size, self.GRID_HEIGHT * self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)
        
        # Draw gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type == 0: continue
                
                # Handle swap animation visualization
                is_swapping = False
                if self.swap_animation:
                    if (c, r) == self.swap_animation["pos1"]:
                        target_c, target_r = self.swap_animation["pos2"]
                        is_swapping = True
                    elif (c, r) == self.swap_animation["pos2"]:
                        target_c, target_r = self.swap_animation["pos1"]
                        is_swapping = True

                if is_swapping:
                    # For auto_advance=False, we can't show smooth interpolation.
                    # We show the gems in their swapped positions to give feedback.
                    pos_x = self.grid_offset_x + target_c * self.cell_size + self.cell_size / 2
                    pos_y = self.grid_offset_y + target_r * self.cell_size + self.cell_size / 2
                else:
                    pos_x = self.grid_offset_x + c * self.cell_size + self.cell_size / 2
                    pos_y = self.grid_offset_y + r * self.cell_size + self.cell_size / 2
                
                self._draw_gem(int(pos_x), int(pos_y), gem_type)

        # Draw grid lines over gems
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_offset_y), (x, grid_rect.bottom))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_offset_x, y), (grid_rect.right, y))

        # Draw selections
        if self.selected_gem:
            sx, sy = self.selected_gem
            rect = pygame.Rect(self.grid_offset_x + sx * self.cell_size, self.grid_offset_y + sy * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3, border_radius=4)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.grid_offset_x + cx * self.cell_size, self.grid_offset_y + cy * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=4)
        
        # Draw particles
        for p in self.particles:
            size = max(0, int(p["life"] / 6))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, p["color"])

    def _draw_gem(self, x, y, gem_type):
        color = self.GEM_COLORS[gem_type - 1]
        shadow_color = tuple(c * 0.5 for c in color)
        highlight_color = tuple(min(255, c * 1.5) for c in color)
        
        pygame.gfxdraw.filled_circle(self.screen, x, y + 2, self.gem_radius, shadow_color)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.gem_radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.gem_radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x - self.gem_radius // 3, y - self.gem_radius // 3, self.gem_radius // 3, highlight_color)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.UI_AREA_HEIGHT / 2))
        self.screen.blit(score_text, score_rect)
        
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(midleft=(20, self.UI_AREA_HEIGHT / 2))
        self.screen.blit(moves_text, moves_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "You Win!" if self.score >= self.TARGET_SCORE else "Game Over"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    game_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # no-op, buttons released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Only step if an action is taken, to simulate turn-based play
        if action != [0, env.prev_space_held, env.prev_shift_held]:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

            if terminated or truncated:
                print(f"--- GAME OVER ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
                
                # Render final state
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                game_screen.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(3000)

                # Reset for new game
                obs, info = env.reset()
                total_reward = 0
        else:
            # If no action, just update the rendering with the current obs
            obs = env._get_observation()

        # Render to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        pygame.display.flip()
            
        clock.tick(15) # Limit frame rate of manual play
        
    env.close()