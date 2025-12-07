import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space while moving to push crystals. Match the pattern on the right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based puzzle game. Navigate the cavern and push crystals into their target locations before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Colors and Visuals ---
    COLOR_BG = (25, 20, 35)
    COLOR_GRID = (50, 45, 65)
    COLOR_WALL = (15, 12, 25)
    COLOR_AVATAR = (100, 150, 255)
    COLOR_AVATAR_GLOW = (50, 80, 200)
    CRYSTAL_COLORS = {
        'red': ((255, 80, 80), (200, 40, 40)),
        'green': ((80, 255, 80), (40, 200, 40)),
        'yellow': ((255, 255, 80), (200, 200, 40)),
        'purple': ((200, 80, 255), (150, 40, 200)),
        'cyan': ((80, 255, 255), (40, 200, 200)),
    }
    COLOR_TARGET_ALPHA = 64
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables
        self.grid = np.array([])
        self.grid_size = (0, 0)
        self.avatar_pos = (0, 0)
        self.crystals = []
        self.current_level = 1
        self.moves_remaining = 0
        self.last_total_dist = 0
        self.last_move_dir = (0, 0)
        self.level_start_moves = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # This is here to ensure the np_random generator is created before reset
        super().reset(seed=None)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 1
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the game state for the current level."""
        # --- Level Progression ---
        level_idx = self.current_level - 1
        grid_w = min(10, 5 + level_idx // 3)
        grid_h = min(10, 5 + level_idx // 3)
        self.grid_size = (grid_w, grid_h)
        num_crystals = min(10, 3 + level_idx // 2)
        
        # --- Grid and Wall Generation ---
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # --- Procedural Generation (Scramble Method) ---
        valid_coords = []
        for r in range(1, self.grid_size[1] - 1):
            for c in range(1, self.grid_size[0] - 1):
                valid_coords.append((c, r))
        
        self.np_random.shuffle(valid_coords)
        
        # Place targets
        target_coords = [valid_coords.pop() for _ in range(num_crystals)]
        
        # Place avatar
        self.avatar_pos = valid_coords.pop()
        
        # Place crystals on targets initially
        self.crystals = []
        crystal_color_keys = list(self.CRYSTAL_COLORS.keys())
        for i in range(num_crystals):
            pos = target_coords[i]
            self.crystals.append({
                'id': i,
                'pos': pos,
                'target_pos': pos,
                'color_key': crystal_color_keys[i % len(crystal_color_keys)],
                'is_correct': True,
                'pulse_offset': self.np_random.random() * math.pi * 2,
            })
            
        # Scramble the puzzle by making random inverse moves
        scramble_moves = 5 + level_idx * 2
        for _ in range(scramble_moves):
            self._perform_random_scramble_move()
            
        self.level_start_moves = int(scramble_moves * 1.8) + num_crystals * 2
        self.moves_remaining = self.level_start_moves
        self.last_total_dist = self._calculate_total_distance()

    def _perform_random_scramble_move(self):
        """Performs a single random valid move to scramble the board from a solved state."""
        potential_moves = []
        # Check moves for avatar
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if self._is_empty((self.avatar_pos[0] + dx, self.avatar_pos[1] + dy)):
                potential_moves.append(('avatar', (dx, dy)))
        
        # Check pulls for crystals
        for i, crystal in enumerate(self.crystals):
            cx, cy = crystal['pos']
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                # Check if avatar is in position to "pull"
                if (cx + dx, cy + dy) == self.avatar_pos:
                    # Check if space behind crystal is empty
                    if self._is_empty((cx - dx, cy - dy)):
                        potential_moves.append(('crystal', i, (dx, dy)))

        if not potential_moves:
            return

        # FIX: np.random.choice cannot handle a list of tuples of different lengths.
        # Instead, we get a random index and select from the list.
        move_index = self.np_random.integers(len(potential_moves))
        move = potential_moves[move_index]
        
        if move[0] == 'avatar':
            _, (dx, dy) = move
            self.avatar_pos = (self.avatar_pos[0] + dx, self.avatar_pos[1] + dy)
        elif move[0] == 'crystal':
            _, crystal_idx, (dx, dy) = move
            self.avatar_pos = self.crystals[crystal_idx]['pos']
            self.crystals[crystal_idx]['pos'] = (self.crystals[crystal_idx]['pos'][0] - dx, self.crystals[crystal_idx]['pos'][1] - dy)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]  # 0-4
        space_held = action[1] == 1  # Boolean
        
        self.steps += 1
        reward = 0.0
        action_taken = False

        if movement != 0:
            self.moves_remaining -= 1
            action_taken = True
            
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
            dx, dy = move_map[movement]
            self.last_move_dir = (dx, dy)
            
            ax, ay = self.avatar_pos
            target_pos = (ax + dx, ay + dy)

            # Check for crystal at target position
            crystal_at_target = self._get_crystal_at(target_pos)

            if space_held and crystal_at_target is not None:
                # --- PUSH LOGIC ---
                push_pos = (target_pos[0] + dx, target_pos[1] + dy)
                if self._is_empty(push_pos):
                    crystal_at_target['pos'] = push_pos
                    self.avatar_pos = target_pos
            elif crystal_at_target is None and self.grid[target_pos[1], target_pos[0]] == 0:
                # --- MOVE LOGIC ---
                self.avatar_pos = target_pos
            else:
                # Action failed (bumped into wall or crystal without pushing)
                pass
        
        # --- REWARD & STATE UPDATE ---
        if action_taken:
            dist_reward, correct_crystal_reward = self._calculate_incremental_reward()
            reward += float(dist_reward + correct_crystal_reward)
            self.score += float(dist_reward + correct_crystal_reward)

        # --- LEVEL COMPLETION CHECK ---
        if self._check_level_complete():
            reward += 100.0
            self.score += 100.0
            self.current_level += 1
            if self.current_level > 15:
                self.game_over = True
            else:
                self._setup_level()

        # --- TERMINATION CHECK ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # Ran out of moves
            reward -= 100.0
            self.score -= 100.0
            self.game_over = True
        
        if self.steps >= 1500:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_crystal_at(self, pos):
        for crystal in self.crystals:
            if crystal['pos'] == pos:
                return crystal
        return None

    def _is_empty(self, pos):
        if not (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]):
            return False
        if self.grid[pos[1], pos[0]] == 1:
            return False
        if self._get_crystal_at(pos) is not None:
            return False
        return True

    def _calculate_incremental_reward(self):
        # Distance-based reward
        current_total_dist = self._calculate_total_distance()
        dist_change = self.last_total_dist - current_total_dist
        self.last_total_dist = current_total_dist
        
        # Correct placement reward
        correct_crystal_reward = 0
        for crystal in self.crystals:
            is_now_correct = crystal['pos'] == crystal['target_pos']
            if is_now_correct and not crystal['is_correct']:
                correct_crystal_reward += 10
            crystal['is_correct'] = is_now_correct
            
        return dist_change, correct_crystal_reward

    def _calculate_total_distance(self):
        total_dist = 0
        for crystal in self.crystals:
            total_dist += abs(crystal['pos'][0] - crystal['target_pos'][0]) + \
                          abs(crystal['pos'][1] - crystal['target_pos'][1])
        return total_dist

    def _check_level_complete(self):
        return all(c['pos'] == c['target_pos'] for c in self.crystals)

    def _check_termination(self):
        return self.game_over or self.moves_remaining <= 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "moves_remaining": self.moves_remaining,
        }

    def _render_game(self):
        # --- Calculate grid rendering parameters ---
        cell_size = int(min((self.screen.get_width() - 200) / self.grid_size[0], 
                            (self.screen.get_height() - 40) / self.grid_size[1]))
        
        offset_x = 40
        offset_y = (self.screen.get_height() - self.grid_size[1] * cell_size) // 2
        
        # --- Draw grid and walls ---
        for r in range(self.grid_size[1]):
            for c in range(self.grid_size[0]):
                rect = pygame.Rect(offset_x + c * cell_size, offset_y + r * cell_size, cell_size, cell_size)
                if self.grid[r, c] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # --- Draw targets and crystals ---
        for crystal in self.crystals:
            # Draw target location
            tx, ty = crystal['target_pos']
            target_color = self.CRYSTAL_COLORS[crystal['color_key']][0]
            target_rect = pygame.Rect(offset_x + tx * cell_size + cell_size * 0.2, 
                                      offset_y + ty * cell_size + cell_size * 0.2, 
                                      cell_size * 0.6, cell_size * 0.6)
            
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*target_color, self.COLOR_TARGET_ALPHA), shape_surf.get_rect(), border_radius=3)
            self.screen.blit(shape_surf, target_rect.topleft)

            # Draw crystal
            cx, cy = crystal['pos']
            center_x = int(offset_x + cx * cell_size + cell_size / 2)
            center_y = int(offset_y + cy * cell_size + cell_size / 2)
            main_color, glow_color = self.CRYSTAL_COLORS[crystal['color_key']]
            
            pulse = 0
            if crystal['is_correct']:
                pulse = math.sin(self.steps * 0.2 + crystal['pulse_offset']) * 3
            
            self._draw_glowing_crystal(center_x, center_y, int(cell_size * 0.35 + pulse), main_color, glow_color)

        # --- Draw avatar ---
        ax, ay = self.avatar_pos
        center_x = int(offset_x + ax * cell_size + cell_size / 2)
        center_y = int(offset_y + ay * cell_size + cell_size / 2)
        self._draw_glowing_crystal(center_x, center_y, int(cell_size * 0.4), self.COLOR_AVATAR, self.COLOR_AVATAR_GLOW, shape='diamond')

    def _draw_glowing_crystal(self, x, y, radius, main_color, glow_color, shape='circle'):
        # Draw glow
        for i in range(radius, 0, -2):
            alpha = 100 * (1 - i / radius)**2
            pygame.gfxdraw.filled_circle(self.screen, x, y, i + 5, (*glow_color, alpha))
        
        # Draw main shape
        if shape == 'circle':
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, main_color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, main_color)
        elif shape == 'diamond':
            points = [
                (x, y - radius), (x + radius, y),
                (x, y + radius), (x - radius, y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, main_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, main_color)
        
        # Highlight
        pygame.gfxdraw.filled_circle(self.screen, int(x + radius * 0.3), int(y - radius * 0.3), int(radius * 0.2), (255, 255, 255, 128))

    def _render_ui(self):
        # --- Left Panel ---
        level_text = self.font_large.render(f"LEVEL {self.current_level}", True, (200, 200, 220))
        self.screen.blit(level_text, (20, 20))
        
        moves_text = self.font_large.render(f"MOVES", True, (200, 200, 220))
        self.screen.blit(moves_text, (20, 60))
        
        moves_val_text = self.font_huge.render(f"{self.moves_remaining}", True, (255, 255, 255))
        self.screen.blit(moves_val_text, (20, 90))
        
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, (180, 180, 200))
        self.screen.blit(score_text, (20, self.screen.get_height() - 30))

        # --- Right Panel (Target Preview) ---
        preview_w = 150
        preview_h = 150
        preview_x = self.screen.get_width() - preview_w - 20
        preview_y = 20
        
        pygame.draw.rect(self.screen, self.COLOR_WALL, (preview_x, preview_y, preview_w, preview_h), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (preview_x, preview_y, preview_w, preview_h), 1, border_radius=5)
        
        preview_cell_size = min(preview_w / self.grid_size[0], preview_h / self.grid_size[1])
        
        for crystal in self.crystals:
            tx, ty = crystal['target_pos']
            color, _ = self.CRYSTAL_COLORS[crystal['color_key']]
            px = int(preview_x + tx * preview_cell_size + preview_cell_size / 2)
            py = int(preview_y + ty * preview_cell_size + preview_cell_size / 2)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(preview_cell_size * 0.4), color)
            
        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.current_level > 15:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
                
            end_text = self.font_huge.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)
            
            score_text = self.font_large.render(f"Final Score: {int(self.score)}", True, (220, 220, 220))
            score_rect = score_text.get_rect(center=(self.screen.get_rect().centerx, self.screen.get_rect().centery + 50))
            self.screen.blit(score_text, score_rect)

# Example of how to run the environment
if __name__ == '__main__':
    # To run with a window, comment out the os.environ line at the top
    # and uncomment the following line
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Event Handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
                movement = 0 # none
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4

                if movement != 0:
                    keys = pygame.key.get_pressed()
                    space_held = 1 if keys[pygame.K_SPACE] else 0
                    shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                    current_action = [movement, space_held, shift_held]
                    action_taken = True

        if action_taken:
            obs, reward, terminated, truncated, info = env.step(current_action)
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                # Game is over, but we let the user see the final screen.
                # A reset is needed to play again (press 'r').

        # --- Rendering ---
        frame = env._get_observation()
        # The observation is (H, W, C) = (400, 640, 3). Pygame needs (W, H).
        # And the surfarray needs to be transposed back.
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    pygame.quit()