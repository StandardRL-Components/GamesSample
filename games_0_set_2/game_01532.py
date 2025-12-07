
# Generated: 2025-08-27T17:26:26.678098
# Source Brief: brief_01532.md
# Brief Index: 1532

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to select a block and trigger a match."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match 3 or more adjacent colored blocks to clear them. Create large combos to reach the target score before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.START_MOVES = 30
        self.TARGET_SCORE = 5000
        self.MAX_EPISODE_STEPS = 1000  # Safety break

        # Calculate grid rendering properties
        self.GRID_AREA_HEIGHT = self.HEIGHT - 40
        self.BLOCK_SIZE = int(min(self.WIDTH / self.GRID_SIZE, self.GRID_AREA_HEIGHT / self.GRID_SIZE) * 0.9)
        self.GRID_WIDTH = self.GRID_SIZE * self.BLOCK_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.BLOCK_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.BLOCK_COLORS = {
            1: (220, 50, 50),   # Red
            2: (50, 220, 50),   # Green
            3: (50, 120, 220),  # Blue
            4: (220, 220, 50),  # Yellow
            5: (180, 50, 220),  # Purple
            6: (100, 110, 120)  # Grey (Obstacle)
        }
        self.NUM_PLAYABLE_COLORS = 5 # 1 through 5

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_popup = pygame.font.SysFont("Arial", 24, bold=True)

        # --- Internal State Variables (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.last_space_held = False
        self.particles = []
        self.score_popups = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_left = self.START_MOVES
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_space_held = False
        self.particles = []
        self.score_popups = []

        self._create_board()
        self._ensure_match_possible()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        movement, space_held, _ = action
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        # 1. Handle Cursor Movement
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        # 2. Handle Block Selection
        if space_pressed and not self.game_over:
            x, y = self.cursor_pos
            color_to_match = self.grid[y, x]

            if color_to_match == 0 or color_to_match > self.NUM_PLAYABLE_COLORS: # Cannot match empty or grey
                reward -= 0.1
                # sound: invalid_move.wav
            else:
                match_group = self._find_match_group(x, y)
                
                if len(match_group) >= 3:
                    # --- Successful Match ---
                    # sound: match_success.wav
                    self.moves_left -= 1
                    
                    # Process this match and any subsequent chain reactions
                    chain_level = 1
                    while len(match_group) >= 3:
                        points, match_reward = self._calculate_score_and_reward(len(match_group), chain_level)
                        self.score += points
                        reward += match_reward
                        reward += len(match_group) # Per-block reward

                        # Create visual effects
                        self._create_particles_for_group(match_group)
                        if points > 0:
                            self._create_score_popup(f"+{points}", match_group[0])

                        # Clear blocks and apply gravity
                        for gx, gy in match_group:
                            self.grid[gy, gx] = 0 # 0 for empty
                        
                        self._apply_gravity_and_refill()
                        
                        # Find next chain reaction
                        chain_level += 1
                        match_group = self._find_all_chains()
                        if len(match_group) >= 3:
                            # sound: chain_reaction.wav
                            pass
                    
                    self._ensure_match_possible()

                else:
                    # --- Failed Match ---
                    reward -= 0.1
                    # sound: invalid_move.wav

        # 3. Check Termination Conditions
        terminated = False
        if self.score >= self.TARGET_SCORE:
            terminated = True
            self.game_over = True
            reward += 100 # Goal-oriented reward for winning
        elif self.moves_left <= 0:
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

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

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "steps": self.steps,
        }
        
    # --- Game Logic Helpers ---
    def _create_board(self):
        self.grid = self.np_random.integers(1, self.NUM_PLAYABLE_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        # Add a few grey blocks for challenge
        for _ in range(self.np_random.integers(2, 5)):
             x, y = self.np_random.integers(0, self.GRID_SIZE, size=2)
             self.grid[y, x] = 6

    def _ensure_match_possible(self):
        while not self._check_for_any_match():
            self._create_board() # Re-create board if no matches are possible

    def _check_for_any_match(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = self.grid[y, x]
                if color == 0 or color > self.NUM_PLAYABLE_COLORS: continue
                # Check horizontal
                if x < self.GRID_SIZE - 2 and self.grid[y, x+1] == color and self.grid[y, x+2] == color:
                    return True
                # Check vertical
                if y < self.GRID_SIZE - 2 and self.grid[y+1, x] == color and self.grid[y+2, x] == color:
                    return True
        return False

    def _find_match_group(self, start_x, start_y):
        color_to_match = self.grid[start_y, start_x]
        if color_to_match == 0 or color_to_match > self.NUM_PLAYABLE_COLORS:
            return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        match_group = []

        while q:
            x, y = q.popleft()
            match_group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                    if self.grid[ny, nx] == color_to_match:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        
        return match_group

    def _find_all_chains(self):
        matches = set()
        visited = set()
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) in visited: continue
                color = self.grid[y, x]
                if color == 0 or color > self.NUM_PLAYABLE_COLORS: continue
                
                # Horizontal check
                h_group = [(x + i, y) for i in range(self.GRID_SIZE) if x + i < self.GRID_SIZE and self.grid[y, x + i] == color]
                if len(h_group) >= 3:
                    for pos in h_group:
                        matches.add(pos)
                        visited.add(pos)

                # Vertical check
                v_group = [(x, y + i) for i in range(self.GRID_SIZE) if y + i < self.GRID_SIZE and self.grid[y + i, x] == color]
                if len(v_group) >= 3:
                    for pos in v_group:
                        matches.add(pos)
                        visited.add(pos)
        return list(matches)


    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_SIZE):
            empty_count = 0
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y, x] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[y + empty_count, x] = self.grid[y, x]
                    self.grid[y, x] = 0
            # Refill top
            for y in range(empty_count):
                self.grid[y, x] = self.np_random.integers(1, self.NUM_PLAYABLE_COLORS + 1)
    
    def _calculate_score_and_reward(self, num_blocks, chain_level):
        base_score = 0
        base_reward = 0
        if num_blocks == 3:
            base_score = 100
            base_reward = 10
        elif num_blocks == 4:
            base_score = 200
            base_reward = 20
        elif num_blocks == 5:
            base_score = 500
            base_reward = 50
        else: # 6+
            base_score = 1000
            base_reward = 100
        
        # Apply chain multiplier
        chain_multiplier = 1 + (chain_level - 1) * 0.5
        return int(base_score * chain_multiplier), base_reward * chain_multiplier

    # --- Rendering Helpers ---
    def _render_game(self):
        self._render_grid_and_blocks()
        self._render_particles_and_effects()
        self._render_cursor()

    def _render_grid_and_blocks(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_id = self.grid[y, x]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.BLOCK_SIZE,
                    self.GRID_OFFSET_Y + y * self.BLOCK_SIZE,
                    self.BLOCK_SIZE, self.BLOCK_SIZE
                )
                
                # Draw grid background
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                if color_id > 0:
                    base_color = self.BLOCK_COLORS[color_id]
                    shadow_color = tuple(max(0, c - 40) for c in base_color)
                    highlight_color = tuple(min(255, c + 40) for c in base_color)
                    
                    # Beveled effect
                    pygame.draw.rect(self.screen, shadow_color, rect)
                    inner_rect = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, base_color, inner_rect)
                    
                    # Highlight
                    pygame.draw.line(self.screen, highlight_color, inner_rect.topleft, inner_rect.topright, 2)
                    pygame.draw.line(self.screen, highlight_color, inner_rect.topleft, inner_rect.bottomleft, 2)


    def _render_cursor(self):
        x, y = self.cursor_pos
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 4 # 0 to 4
        rect = pygame.Rect(
            self.GRID_OFFSET_X + x * self.BLOCK_SIZE - 2 + pulse/2,
            self.GRID_OFFSET_Y + y * self.BLOCK_SIZE - 2 + pulse/2,
            self.BLOCK_SIZE + 4 - pulse,
            self.BLOCK_SIZE + 4 - pulse
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=4)


    def _render_particles_and_effects(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] -= 0.2
            if p['life'] <= 0 or p['size'] <= 0:
                self.particles.remove(p)
            else:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]),
                    max(0, int(p['size'])),
                    (*p['color'], p['life'] * 4) # Fade out
                )

        # Update and draw score popups
        for s in self.score_popups[:]:
            s['pos'][1] -= 0.5 # Move up
            s['life'] -= 1
            if s['life'] <= 0:
                self.score_popups.remove(s)
            else:
                text_surf = self.font_popup.render(s['text'], True, (*s['color'], s['life'] * 3))
                text_rect = text_surf.get_rect(center=(int(s['pos'][0]), int(s['pos'][1])))
                self.screen.blit(text_surf, text_rect)


    def _create_particles_for_group(self, group):
        for x, y in group:
            center_x = self.GRID_OFFSET_X + x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
            center_y = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
            color = self.BLOCK_COLORS[self.grid[y,x]] if self.grid[y,x] > 0 else (200,200,200)

            for _ in range(10): # 10 particles per block
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                self.particles.append({
                    'pos': [center_x, center_y],
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'size': random.uniform(3, 7),
                    'color': color,
                    'life': random.randint(30, 60)
                })

    def _create_score_popup(self, text, grid_pos):
        x, y = grid_pos
        center_x = self.GRID_OFFSET_X + x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        center_y = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        self.score_popups.append({
            'text': text,
            'pos': [center_x, center_y],
            'color': (255, 255, 100),
            'life': 60 # 2 seconds at 30fps
        })

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Moves
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 10, 5))
        self.screen.blit(moves_text, moves_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.score >= self.TARGET_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)


    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == "__main__":
    # To play the game manually, run this file.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Grid")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op, no press
        
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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # If a key was pressed, take a step
        if action.any():
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
            if terminated:
                print("Game Over!")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()