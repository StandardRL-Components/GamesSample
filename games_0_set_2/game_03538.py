
# Generated: 2025-08-27T23:40:14.827352
# Source Brief: brief_03538.md
# Brief Index: 3538

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a gem, "
        "then move to an adjacent gem and press space again to swap. Press shift to reshuffle the board (costs a move)."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more of the same type. "
        "Create combos and chain reactions to maximize your score. Collect 100 gems in 20 moves to win!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.GRID_WIDTH = 8
        self.GRID_HEIGHT = 8
        self.NUM_GEM_TYPES = 6
        self.MAX_MOVES = 20
        self.WIN_GEMS = 100
        self.MAX_STEPS = 1000

        # Screen dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_AREA_HEIGHT = 360
        self.UI_AREA_HEIGHT = 40
        self.GEM_SIZE = self.GRID_AREA_HEIGHT // self.GRID_HEIGHT
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.GEM_SIZE) // 2
        self.GRID_OFFSET_Y = self.UI_AREA_HEIGHT

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
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.GEM_COLORS = [
            (255, 50, 50),   # Red
            (50, 255, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 255, 50),  # Yellow
            (255, 50, 255),  # Magenta
            (50, 255, 255),  # Cyan
        ]

        # Game state variables
        self.grid = None
        self.cursor_pos = None
        self.first_selection = None
        self.moves_remaining = 0
        self.gems_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.space_was_held = False
        self.shift_was_held = False
        self.particles = []
        self.animation_state = 'IDLE'
        self.animation_data = {}
        self.animation_timer = 0.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_remaining = self.MAX_MOVES
        self.gems_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.first_selection = None
        self.space_was_held = True # Prevent action on first frame
        self.shift_was_held = True
        
        self.particles = []
        self.animation_state = 'IDLE'
        self.animation_data = {}
        self.animation_timer = 0.0

        self._generate_initial_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held
        self.space_was_held = space_held
        self.shift_was_held = shift_held

        # Animate existing state changes before processing new actions
        if self.animation_state != 'IDLE':
            self._update_animations()
        else:
            # Process player input
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

            if shift_pressed and self.moves_remaining > 0:
                self.moves_remaining -= 1
                reward = -5.0 # Penalty for reshuffling
                self._shuffle_board()
                self.first_selection = None
            
            elif space_pressed:
                reward = self._handle_selection()
        
        # Check for game termination
        if self.gems_collected >= self.WIN_GEMS and not self.game_over:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100
        elif self.moves_remaining <= 0 and self.animation_state == 'IDLE' and not self.game_over:
            self.game_over = True
            self.game_won = False
            terminated = True
            reward += -50
        
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True
            reward += -50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_selection(self):
        cx, cy = self.cursor_pos
        if self.first_selection is None:
            self.first_selection = [cx, cy]
            # Sound: select_gem.wav
            return 0.0
        else:
            fx, fy = self.first_selection
            is_adjacent = abs(cx - fx) + abs(cy - fy) == 1
            if is_adjacent:
                return self._attempt_swap(self.first_selection, self.cursor_pos)
            else:
                self.first_selection = None # Cancel selection
                # Sound: cancel_selection.wav
                return 0.0
    
    def _attempt_swap(self, pos1, pos2):
        self.moves_remaining -= 1
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Perform swap
        self.grid[y1][x1], self.grid[y2][x2] = self.grid[y2][x2], self.grid[y1][x1]
        
        matches1 = self._find_matches_at_pos(x1, y1)
        matches2 = self._find_matches_at_pos(x2, y2)
        
        all_matches = set(matches1) | set(matches2)

        if not all_matches:
            # Invalid swap, swap back
            self.grid[y1][x1], self.grid[y2][x2] = self.grid[y2][x2], self.grid[y1][x1]
            self.moves_remaining += 1
            self.first_selection = None
            # Sound: invalid_swap.wav
            return -0.1
        else:
            # Valid swap, start animation and processing chain
            self.animation_state = 'MATCH'
            self.animation_data = {'matches': all_matches, 'chain': 0}
            self.first_selection = None
            # Sound: match_start.wav
            return 0.0 # Reward is given during animation processing

    def _update_animations(self):
        if self.animation_state == 'MATCH':
            reward = self._process_matches()
            self.animation_state = 'FALL'
            self.animation_data = self._apply_gravity_and_fill()
            # This is a bit of a hack to inject reward from an async process
            # In a real game, this might be queued. Here we just add it.
            if self.steps > 0: self.score += int(reward) # Avoid changing score on reset
        elif self.animation_state == 'FALL':
            # After falling, check for new matches (chain reaction)
            new_matches = self._find_all_matches()
            if new_matches:
                self.animation_state = 'MATCH'
                self.animation_data['matches'] = new_matches
                self.animation_data['chain'] += 1
                # Sound: chain_reaction.wav
            else:
                self.animation_state = 'IDLE'
                self.animation_data = {}
                if not self._find_possible_moves():
                    self._shuffle_board()

    def _process_matches(self):
        matches = self.animation_data['matches']
        chain_level = self.animation_data['chain']
        
        num_cleared = len(matches)
        self.gems_collected += num_cleared
        
        # Base reward: 1 per gem
        reward = num_cleared
        
        # Combo bonus
        if num_cleared == 4: reward += 5
        elif num_cleared >= 5: reward += 10
        
        # Chain bonus
        reward *= (1 + chain_level * 0.5)

        for x, y in matches:
            self._create_particles(x, y, self.grid[y][x])
            self.grid[y][x] = -1 # Mark as empty
        
        # Sound: gems_cleared.wav
        return reward

    def _apply_gravity_and_fill(self):
        fall_data = []
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] != -1:
                    if y != empty_row:
                        self.grid[empty_row][x] = self.grid[y][x]
                        self.grid[y][x] = -1
                        fall_data.append({'from': (x, y), 'to': (x, empty_row), 'type': self.grid[empty_row][x]})
                    empty_row -= 1
            # Fill new gems
            for y in range(empty_row, -1, -1):
                self.grid[y][x] = self.np_random.integers(0, self.NUM_GEM_TYPES)
                fall_data.append({'from': (x, y - (empty_row + 1)), 'to': (x, y), 'type': self.grid[y][x]})
        return fall_data

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH * self.GEM_SIZE, self.GRID_HEIGHT * self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[y][x]
                if gem_type != -1:
                    self._draw_gem(x, y, gem_type)
        
        # Draw particles
        self._update_and_draw_particles()

        # Draw cursor
        if not self.game_over:
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.GEM_SIZE, self.GRID_OFFSET_Y + cy * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
            
            # Draw selected highlight
            if self.first_selection:
                fx, fy = self.first_selection
                selected_rect = pygame.Rect(self.GRID_OFFSET_X + fx * self.GEM_SIZE, self.GRID_OFFSET_Y + fy * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_SELECTED, selected_rect, 4, border_radius=8)

            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)
    
    def _draw_gem(self, x, y, gem_type):
        center_x = self.GRID_OFFSET_X + int((x + 0.5) * self.GEM_SIZE)
        center_y = self.GRID_OFFSET_Y + int((y + 0.5) * self.GEM_SIZE)
        radius = int(self.GEM_SIZE * 0.4)
        
        color = self.GEM_COLORS[gem_type % len(self.GEM_COLORS)]
        highlight = tuple(min(255, c + 60) for c in color)
        shadow = tuple(max(0, c - 60) for c in color)
        
        # Draw shadow, base, and highlight for a 3D effect
        pygame.gfxdraw.filled_circle(self.screen, center_x + 2, center_y + 2, radius, shadow)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, center_x - 2, center_y - 2, int(radius * 0.8), highlight)

    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, (10, 20, 30), (0, 0, self.SCREEN_WIDTH, self.UI_AREA_HEIGHT))
        
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 8))
        
        # Moves
        moves_text = f"MOVES: {self.moves_remaining}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH // 2 - moves_surf.get_width() // 2, 8))
        
        # Gems
        gems_text = f"GEMS: {self.gems_collected} / {self.WIN_GEMS}"
        gems_surf = self.font_ui.render(gems_text, True, self.COLOR_TEXT)
        self.screen.blit(gems_surf, (self.SCREEN_WIDTH - gems_surf.get_width() - 10, 8))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            text_surf = self.font_game_over.render(end_text, True, color)
            self.screen.blit(text_surf, (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - text_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "gems_collected": self.gems_collected,
        }

    # --- Helper Functions ---
    
    def _generate_initial_board(self):
        self.grid = [[0] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self.grid[y][x] = self.np_random.integers(0, self.NUM_GEM_TYPES)
        
        while self._find_all_matches() or not self._find_possible_moves():
            self._shuffle_board()

    def _shuffle_board(self):
        flat_grid = [self.grid[y][x] for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH)]
        self.np_random.shuffle(flat_grid)
        self.grid = [[flat_grid[y * self.GRID_WIDTH + x] for x in range(self.GRID_WIDTH)] for y in range(self.GRID_HEIGHT)]
        
        while self._find_all_matches():
            matches = self._find_all_matches()
            for x, y in matches:
                possible_types = list(range(self.NUM_GEM_TYPES))
                if x > 0 and self.grid[y][x-1] in possible_types: possible_types.remove(self.grid[y][x-1])
                if y > 0 and self.grid[y-1][x] in possible_types: possible_types.remove(self.grid[y-1][x])
                self.grid[y][x] = self.np_random.choice(possible_types) if possible_types else 0

    def _find_matches_at_pos(self, x, y):
        gem_type = self.grid[y][x]
        if gem_type == -1: return set()
        
        # Horizontal check
        h_matches = {(x, y)}
        # Left
        for i in range(x - 1, -1, -1):
            if self.grid[y][i] == gem_type: h_matches.add((i, y))
            else: break
        # Right
        for i in range(x + 1, self.GRID_WIDTH):
            if self.grid[y][i] == gem_type: h_matches.add((i, y))
            else: break
        
        # Vertical check
        v_matches = {(x, y)}
        # Up
        for i in range(y - 1, -1, -1):
            if self.grid[i][x] == gem_type: v_matches.add((x, i))
            else: break
        # Down
        for i in range(y + 1, self.GRID_HEIGHT):
            if self.grid[i][x] == gem_type: v_matches.add((x, i))
            else: break
            
        final_matches = set()
        if len(h_matches) >= 3: final_matches.update(h_matches)
        if len(v_matches) >= 3: final_matches.update(v_matches)
        return final_matches

    def _find_all_matches(self):
        all_matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                all_matches.update(self._find_matches_at_pos(x, y))
        return all_matches

    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Check swap right
                if x < self.GRID_WIDTH - 1:
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                    if self._find_matches_at_pos(x, y) or self._find_matches_at_pos(x+1, y):
                        self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                        return True
                    self.grid[y][x], self.grid[y][x+1] = self.grid[y][x+1], self.grid[y][x]
                # Check swap down
                if y < self.GRID_HEIGHT - 1:
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                    if self._find_matches_at_pos(x, y) or self._find_matches_at_pos(x, y+1):
                        self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
                        return True
                    self.grid[y][x], self.grid[y+1][x] = self.grid[y+1][x], self.grid[y][x]
        return False

    def _create_particles(self, grid_x, grid_y, gem_type):
        center_x = self.GRID_OFFSET_X + (grid_x + 0.5) * self.GEM_SIZE
        center_y = self.GRID_OFFSET_Y + (grid_y + 0.5) * self.GEM_SIZE
        color = self.GEM_COLORS[gem_type % len(self.GEM_COLORS)]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'radius': random.uniform(2, 5),
                'life': random.randint(20, 40),
                'color': color
            })

    def _update_and_draw_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['radius'] -= 0.1
            p['life'] -= 1
            if p['radius'] <= 0 or p['life'] <= 0:
                self.particles.remove(p)
            else:
                pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, p['radius']))
    
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0
        space_held = False
        shift_held = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("Game Over!")
            print(f"Final Score: {info['score']}, Gems: {info['gems_collected']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS for human play
        
    pygame.quit()