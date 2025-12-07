
# Generated: 2025-08-28T06:01:18.053223
# Source Brief: brief_02805.md
# Brief Index: 2805

        
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
        "Controls: Arrow keys to move cursor. Space to select a gem, then move the cursor to an adjacent gem and press Space again to swap. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap colorful gems to create matches of 3 or more. Plan your moves to create chain reactions and reach the target score before you run out of turns!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.GEM_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.GEM_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.GEM_SIZE) // 2
        self.WIN_SCORE = 1000
        self.STARTING_MOVES = 50
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (30, 40, 55)
        self.COLOR_GRID_LINE = (50, 60, 75)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.GEM_COLORS = [
            (255, 50, 50),   # Red
            (50, 255, 50),   # Green
            (80, 80, 255),   # Blue
            (255, 255, 50),  # Yellow
            (200, 50, 255),  # Purple
            (255, 150, 50),  # Orange
        ]

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.steps = 0
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.win = False
        self.last_match_effects = []
        self.last_swap_invalid = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.STARTING_MOVES
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        self.last_match_effects = []
        self.last_swap_invalid = False
        
        self._generate_initial_board()
        
        return self._get_observation(), self._get_info()

    def _generate_initial_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_all_matches():
                break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Clear transient effects from the last step
        self.last_match_effects = []
        self.last_swap_invalid = False
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # 1. Handle player input
        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)

        if shift_pressed:
            self.selected_pos = None
        
        swap_attempted = False
        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            if self.selected_pos is None:
                self.selected_pos = cursor_tuple
            else:
                if self.selected_pos == cursor_tuple:
                    self.selected_pos = None
                elif self._are_adjacent(self.selected_pos, cursor_tuple):
                    swap_attempted = True
                else:
                    self.selected_pos = cursor_tuple
        
        # 2. If a swap is triggered, resolve the entire turn
        if swap_attempted:
            self.moves_remaining -= 1
            p1 = self.selected_pos
            p2 = tuple(self.cursor_pos)

            self._swap_gems(p1, p2)
            
            matches = self._find_all_matches()
            if not matches:
                # Invalid swap, no match created
                reward = -0.2
                self._swap_gems(p1, p2) # Swap back
                self.last_swap_invalid = True
            else:
                # Valid swap, resolve matches and chains
                total_gems_this_turn = 0
                chain_reaction = False
                
                while matches:
                    if total_gems_this_turn > 0:
                        chain_reaction = True
                    
                    matched_coords = set()
                    for match in matches:
                        for pos in match:
                            matched_coords.add(pos)
                    
                    num_matched = len(matched_coords)
                    total_gems_this_turn += num_matched
                    self.score += num_matched * 10
                    
                    # Store effects for rendering
                    self.last_match_effects.extend(list(matched_coords))
                    
                    # Remove gems
                    for r, c in matched_coords:
                        self.grid[r, c] = 0
                    
                    # Sound effect placeholder
                    # play_sound('match')
                    
                    self._collapse_gems()
                    self._refill_gems()
                    
                    matches = self._find_all_matches()

                reward += total_gems_this_turn
                if chain_reaction:
                    reward += 5
                    # play_sound('chain_reaction')

            self.selected_pos = None

        self.steps += 1

        # 3. Check for termination
        terminated = self.score >= self.WIN_SCORE or self.moves_remaining <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                self.win = True
                reward += 100
            else:
                reward += -10

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _swap_gems(self, p1, p2):
        r1, c1 = p1
        r2, c2 = p2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _are_adjacent(self, p1, p2):
        r1, c1 = p1
        r2, c2 = p2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _find_all_matches(self):
        matches = []
        # Horizontal matches
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                gem_type = self.grid[r, c]
                if gem_type > 0 and self.grid[r, c+1] == gem_type and self.grid[r, c+2] == gem_type:
                    match = [(r, c), (r, c+1), (r, c+2)]
                    # Extend match for 4s and 5s
                    for i in range(c + 3, self.GRID_SIZE):
                        if self.grid[r, i] == gem_type:
                            match.append((r, i))
                        else:
                            break
                    matches.append(match)
                    c += len(match) - 1

        # Vertical matches
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                gem_type = self.grid[r, c]
                if gem_type > 0 and self.grid[r+1, c] == gem_type and self.grid[r+2, c] == gem_type:
                    match = [(r, c), (r+1, c), (r+2, c)]
                    for i in range(r + 3, self.GRID_SIZE):
                        if self.grid[i, c] == gem_type:
                            match.append((i, c))
                        else:
                            break
                    matches.append(match)
                    r += len(match) - 1
        return matches

    def _collapse_gems(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _refill_gems(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_SIZE * self.GEM_SIZE, self.GRID_SIZE * self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)
        
        # Draw gems and match effects
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type > 0:
                    self._draw_gem(gem_type, r, c)
        
        # Draw match effects
        if self.last_match_effects:
            for r, c in self.last_match_effects:
                self._draw_match_effect(r, c)
        
        # Draw invalid swap effect
        if self.last_swap_invalid:
            if self.selected_pos:
                self._draw_invalid_swap_effect(self.selected_pos)
            self._draw_invalid_swap_effect(tuple(self.cursor_pos))

        # Draw selection and cursor
        if self.selected_pos:
            self._draw_selection(self.selected_pos)
        
        self._draw_cursor()

    def _draw_gem(self, gem_type, r, c):
        x = self.GRID_OFFSET_X + c * self.GEM_SIZE
        y = self.GRID_OFFSET_Y + r * self.GEM_SIZE
        center_x, center_y = x + self.GEM_SIZE // 2, y + self.GEM_SIZE // 2
        
        color = self.GEM_COLORS[gem_type - 1]
        light_color = tuple(min(255, val + 60) for val in color)
        
        radius = self.GEM_SIZE * 0.4
        
        # Draw base shape
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(radius), light_color)

        # Draw unique shape on top
        s = self.GEM_SIZE * 0.2
        if gem_type == 1: # Red - Square
            pygame.draw.rect(self.screen, light_color, (center_x - s/2, center_y - s/2, s, s), 2)
        elif gem_type == 2: # Green - Triangle
            points = [(center_x, center_y - s/1.5), (center_x - s/1.5, center_y + s/2), (center_x + s/1.5, center_y + s/2)]
            pygame.gfxdraw.aapolygon(self.screen, points, light_color)
        elif gem_type == 3: # Blue - Diamond
            points = [(center_x, center_y - s), (center_x - s, center_y), (center_x, center_y + s), (center_x + s, center_y)]
            pygame.gfxdraw.aapolygon(self.screen, points, light_color)
        elif gem_type == 4: # Yellow - Hexagon
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                points.append((center_x + s * math.cos(angle), center_y + s * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, light_color)
        elif gem_type == 5: # Purple - Star
            points = []
            for i in range(10):
                r_star = s if i % 2 == 0 else s * 0.5
                angle = math.pi / 5 * i - math.pi / 2
                points.append((center_x + r_star * math.cos(angle), center_y + r_star * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, light_color)
        elif gem_type == 6: # Orange - X
            pygame.draw.line(self.screen, light_color, (center_x - s/1.5, center_y - s/1.5), (center_x + s/1.5, center_y + s/1.5), 2)
            pygame.draw.line(self.screen, light_color, (center_x - s/1.5, center_y + s/1.5), (center_x + s/1.5, center_y - s/1.5), 2)

    def _draw_cursor(self):
        c, r = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + c * self.GEM_SIZE, self.GRID_OFFSET_Y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
        
        # Pulsing effect
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_surface = pygame.Surface((self.GEM_SIZE, self.GEM_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.GEM_SIZE, self.GEM_SIZE), 4, border_radius=4)
        self.screen.blit(cursor_surface, rect.topleft)
        
    def _draw_selection(self, pos):
        r, c = pos
        rect = pygame.Rect(self.GRID_OFFSET_X + c * self.GEM_SIZE, self.GRID_OFFSET_Y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 4, border_radius=4)

    def _draw_match_effect(self, r, c):
        x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE // 2
        y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE // 2
        radius = self.GEM_SIZE // 2
        # Simple starburst effect
        for i in range(8):
            angle = i * math.pi / 4
            start_pos = (x + radius * 0.5 * math.cos(angle), y + radius * 0.5 * math.sin(angle))
            end_pos = (x + radius * 1.2 * math.cos(angle), y + radius * 1.2 * math.sin(angle))
            pygame.draw.line(self.screen, (255, 255, 150), start_pos, end_pos, 2)

    def _draw_invalid_swap_effect(self, pos):
        r, c = pos
        rect = pygame.Rect(self.GRID_OFFSET_X + c * self.GEM_SIZE, self.GRID_OFFSET_Y + r * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, (255, 0, 0), rect, 3, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = f"Score: {self.score}"
        self._draw_text(score_text, self.font_large, 20, 10)
        
        # Moves
        moves_text = f"Moves: {self.moves_remaining}"
        text_width = self.font_large.size(moves_text)[0]
        self._draw_text(moves_text, self.font_large, self.WIDTH - text_width - 20, 10)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            msg_width, msg_height = self.font_large.size(msg)
            self._draw_text(msg, self.font_large, self.WIDTH/2 - msg_width/2, self.HEIGHT/2 - msg_height/2)
            
    def _draw_text(self, text, font, x, y):
        shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow, (x + 2, y + 2))
        surface = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(surface, (x, y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": list(self.cursor_pos),
            "selected_pos": self.selected_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # To run headlessly (for testing)
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated:
    #         print(f"Game over! Final Info: {info}")
    #         obs, info = env.reset()
    
    # To run with visualization
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, release, release
        
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
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")

        if terminated:
            print(f"Game over! Final Info: {info}")
            # Keep showing the final screen for a moment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()