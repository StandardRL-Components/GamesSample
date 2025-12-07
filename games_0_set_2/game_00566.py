
# Generated: 2025-08-27T14:02:28.153328
# Source Brief: brief_00566.md
# Brief Index: 566

        
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
        "Controls: Arrow keys to move cursor. Space to swap with gem in last moved direction. Shift to reshuffle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzle game. Swap gems to create matches of 3 or more. Plan combos to clear the board!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.GEM_SIZE = 40
        self.GEM_RADIUS = self.GEM_SIZE // 2 - 2
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.GEM_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.GEM_SIZE) // 2
        self.NUM_GEM_TYPES = 6
        self.MAX_MOVES = 50
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SWAP_INDICATOR = (180, 180, 180)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 150, 50)   # Orange
        ]
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 18)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir_idx = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_state = None
        self.animations = None
        self.pending_reward = None
        self.combo_multiplier = None

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.last_move_dir_idx = 1 # Default to Up
        self.game_state = 'IDLE' # States: IDLE, SWAPPING, MATCHING, REFILLING
        self.animations = []
        self.pending_reward = 0
        self.combo_multiplier = 1

        self._generate_initial_grid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.pending_reward = 0

        self._handle_input(action)
        self._update_game_state()
        self._update_animations()

        reward = self.pending_reward
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            self.game_state = 'GAME_OVER'
            if self.moves_left <= 0 and not self._is_board_clear():
                 self.pending_reward -= 10 # Penalty for running out of moves

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_state != 'IDLE':
            return

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Shift action (reshuffle) takes precedence
        if shift_held:
            # Sound: Reshuffle_Sound.play()
            self.moves_left -= 1
            self.pending_reward -= 5
            self._generate_initial_grid() # This also handles finding a valid starting state
            self.animations.append({'type': 'board_flash', 'progress': 0, 'duration': 15})
            return

        # Movement action
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
        if movement in move_map:
            self.last_move_dir_idx = movement
            dx, dy = move_map[movement]
            self.cursor_pos = (
                max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx)),
                max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))
            )

        # Space action (swap)
        if space_held:
            dx, dy = move_map[self.last_move_dir_idx]
            target_pos = (self.cursor_pos[0] + dx, self.cursor_pos[1] + dy)

            if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT:
                self.moves_left -= 1
                self.combo_multiplier = 1
                self._swap_gems(self.cursor_pos, target_pos)
                
                is_match = self._check_for_matches_at([self.cursor_pos, target_pos])

                self.animations.append({
                    'type': 'swap', 'pos1': self.cursor_pos, 'pos2': target_pos,
                    'progress': 0, 'duration': 10, 'is_valid': is_match
                })
                self.game_state = 'SWAPPING'
                # Sound: Swap_Sound.play()

    def _update_game_state(self):
        if self.game_state == 'SWAPPING' and not self._is_animating('swap'):
            # Swap animation finished, check if it was invalid
            last_swap = next((a for a in self.animations if a.get('was_swap')), None)
            if last_swap and not last_swap['is_valid']:
                # Sound: Invalid_Swap_Sound.play()
                self._swap_gems(last_swap['pos1'], last_swap['pos2']) # Swap back
                self.pending_reward -= 0.1
                self.game_state = 'IDLE'
            else:
                self.game_state = 'MATCHING'

        if self.game_state == 'MATCHING':
            matches = self._find_all_matches()
            if matches:
                # Sound: Match_Found_Sound.play()
                num_matched = len(matches)
                base_reward = num_matched
                combo_bonus = (num_matched - 2) * self.combo_multiplier
                self.pending_reward += base_reward + combo_bonus

                for x, y in matches:
                    self._add_particles(x, y, self.grid[y][x])
                    self.grid[y][x] = -1 # Mark for removal
                
                self.animations.append({'type': 'shrink', 'gems': list(matches), 'progress': 0, 'duration': 8})
                self.game_state = 'REFILLING'
                self.combo_multiplier += 1
            else:
                # No more matches, check for possible moves
                if not self._check_for_any_moves():
                    # Sound: No_Moves_Sound.play()
                    self.moves_left -= 2 # Penalty for auto-shuffle
                    self.pending_reward -= 2
                    self._generate_initial_grid()
                    self.animations.append({'type': 'board_flash', 'progress': 0, 'duration': 15, 'color': self.GEM_COLORS[0]})
                self.game_state = 'IDLE'

        if self.game_state == 'REFILLING' and not self._is_animating('shrink'):
            self._handle_gravity()
            self._fill_new_gems()
            self.animations.append({'type': 'fall', 'progress': 0, 'duration': 12})
            self.game_state = 'MATCHING' # Check for new chain-reaction matches

    def _update_animations(self):
        active_animations = []
        for anim in self.animations:
            anim['progress'] += 1
            if anim['progress'] < anim['duration']:
                active_animations.append(anim)
            elif anim['type'] == 'swap':
                anim['was_swap'] = True # Mark as finished for logic check
                active_animations.append(anim)
                # Keep it for one more frame so logic can see it finished
            elif anim.get('was_swap'):
                pass # Now remove it
        self.animations = active_animations

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_game(self):
        # Draw grid background
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(self.GRID_OFFSET_X + x * self.GEM_SIZE,
                                   self.GRID_OFFSET_Y + y * self.GEM_SIZE,
                                   self.GEM_SIZE, self.GEM_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw gems
        animated_gems = set()
        for anim in self.animations:
            if anim['type'] == 'swap':
                p = anim['progress'] / anim['duration']
                p = (1 - math.cos(p * math.pi)) / 2 # Ease in/out
                
                pos1_x, pos1_y = self._grid_to_pixel(anim['pos1'][0], anim['pos1'][1])
                pos2_x, pos2_y = self._grid_to_pixel(anim['pos2'][0], anim['pos2'][1])

                gem1_type = self.grid[anim['pos2'][1]][anim['pos2'][0]]
                gem2_type = self.grid[anim['pos1'][1]][anim['pos1'][0]]

                curr1_x = int(pos1_x + (pos2_x - pos1_x) * p)
                curr1_y = int(pos1_y + (pos2_y - pos1_y) * p)
                self._draw_gem(curr1_x, curr1_y, gem1_type)

                curr2_x = int(pos2_x + (pos1_x - pos2_x) * p)
                curr2_y = int(pos2_y + (pos1_y - pos2_y) * p)
                self._draw_gem(curr2_x, curr2_y, gem2_type)

                animated_gems.add(anim['pos1'])
                animated_gems.add(anim['pos2'])

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) in animated_gems or self.grid[y][x] == -1:
                    continue
                
                gem_type = self.grid[y][x]
                px, py = self._grid_to_pixel(x, y)
                
                # Handle shrink animation
                shrink_anim = next((a for a in self.animations if a['type'] == 'shrink' and (x, y) in a['gems']), None)
                if shrink_anim:
                    p = shrink_anim['progress'] / shrink_anim['duration']
                    radius = int(self.GEM_RADIUS * (1 - p))
                    self._draw_gem(px, py, gem_type, radius)
                    continue

                # Handle fall animation
                fall_anim = next((a for a in self.animations if a['type'] == 'fall'), None)
                if fall_anim and 'fall_dist' in self.grid[y][x]:
                    p = fall_anim['progress'] / fall_anim['duration']
                    p = (1 - math.cos(p * math.pi)) / 2 # Ease in/out
                    start_y = y - self.grid[y][x]['fall_dist']
                    _, start_py = self._grid_to_pixel(x, start_y)
                    draw_py = int(start_py + (py - start_py) * p)
                    self._draw_gem(px, draw_py, self.grid[y][x]['type'])
                    continue

                if isinstance(self.grid[y][x], dict):
                     self._draw_gem(px, py, self.grid[y][x]['type'])
                else:
                     self._draw_gem(px, py, self.grid[y][x])
        
        self._render_particles()
        self._render_cursor()
        
        board_flash = next((a for a in self.animations if a['type'] == 'board_flash'), None)
        if board_flash:
            p = board_flash['progress'] / board_flash['duration']
            alpha = int(128 * math.sin(p * math.pi))
            flash_surface = pygame.Surface((self.GRID_WIDTH * self.GEM_SIZE, self.GRID_HEIGHT * self.GEM_SIZE), pygame.SRCALPHA)
            color = board_flash.get('color', (255, 255, 255))
            flash_surface.fill((*color, alpha))
            self.screen.blit(flash_surface, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))

        # Game Over
        if self.game_state == 'GAME_OVER':
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            win_text = "BOARD CLEARED!" if self._is_board_clear() else "GAME OVER"
            text = self.font_main.render(win_text, True, self.COLOR_CURSOR)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {int(self.score)}", True, self.COLOR_CURSOR)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, final_score_rect)
            
    def _render_cursor(self):
        if self.game_state != 'IDLE': return

        # Main cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.GEM_SIZE,
                           self.GRID_OFFSET_Y + cy * self.GEM_SIZE,
                           self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=5)

        # Swap indicator
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = move_map[self.last_move_dir_idx]
        tx, ty = cx + dx, cy + dy
        if 0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT:
            rect = pygame.Rect(self.GRID_OFFSET_X + tx * self.GEM_SIZE,
                               self.GRID_OFFSET_Y + ty * self.GEM_SIZE,
                               self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SWAP_INDICATOR, rect, 1, border_radius=5)

    def _render_particles(self):
        for p in self.animations:
            if p['type'] == 'particle':
                p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
                p['vel'] = (p['vel'][0], p['vel'][1] + 0.1) # Gravity
                p['radius'] -= 0.2
                if p['radius'] > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def _draw_gem(self, px, py, gem_type, radius=None):
        if gem_type < 0 or gem_type >= self.NUM_GEM_TYPES: return
        if radius is None: radius = self.GEM_RADIUS
        if radius <= 0: return

        color = self.GEM_COLORS[gem_type]
        shadow_color = (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50))
        highlight_color = (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50))
        
        # Draw a simple circle gem with a highlight
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, shadow_color)
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius-1, color)
        pygame.gfxdraw.filled_circle(self.screen, px-radius//3, py-radius//3, radius//3, highlight_color)

    def _grid_to_pixel(self, x, y):
        return (self.GRID_OFFSET_X + x * self.GEM_SIZE + self.GEM_SIZE // 2,
                self.GRID_OFFSET_Y + y * self.GEM_SIZE + self.GEM_SIZE // 2)

    def _generate_initial_grid(self):
        while True:
            self.grid = [[self.np_random.integers(0, self.NUM_GEM_TYPES) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
            if not self._find_all_matches() and self._check_for_any_moves():
                break

    def _swap_gems(self, pos1, pos2):
        self.grid[pos1[1]][pos1[0]], self.grid[pos2[1]][pos2[0]] = \
            self.grid[pos2[1]][pos2[0]], self.grid[pos1[1]][pos1[0]]

    def _find_all_matches(self):
        to_remove = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[y][x] == self.grid[y][x+1] == self.grid[y][x+2] and self.grid[y][x] != -1:
                    to_remove.update([(x, y), (x+1, y), (x+2, y)])
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[y][x] == self.grid[y+1][x] == self.grid[y+2][x] and self.grid[y][x] != -1:
                    to_remove.update([(x, y), (x, y+1), (x, y+2)])
        return to_remove

    def _check_for_matches_at(self, positions):
        for x, y in positions:
            gem_type = self.grid[y][x]
            if gem_type == -1: continue
            # Check horizontal
            if x > 0 and x < self.GRID_WIDTH - 1 and self.grid[y][x-1] == gem_type and self.grid[y][x+1] == gem_type: return True
            if x > 1 and self.grid[y][x-1] == gem_type and self.grid[y][x-2] == gem_type: return True
            if x < self.GRID_WIDTH - 2 and self.grid[y][x+1] == gem_type and self.grid[y][x+2] == gem_type: return True
            # Check vertical
            if y > 0 and y < self.GRID_HEIGHT - 1 and self.grid[y-1][x] == gem_type and self.grid[y+1][x] == gem_type: return True
            if y > 1 and self.grid[y-1][x] == gem_type and self.grid[y-2][x] == gem_type: return True
            if y < self.GRID_HEIGHT - 2 and self.grid[y+1][x] == gem_type and self.grid[y+2][x] == gem_type: return True
        return False
        
    def _check_for_any_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Check swap right
                if x < self.GRID_WIDTH - 1:
                    self._swap_gems((x, y), (x+1, y))
                    if self._check_for_matches_at([(x, y), (x+1, y)]):
                        self._swap_gems((x, y), (x+1, y)) # Swap back
                        return True
                    self._swap_gems((x, y), (x+1, y)) # Swap back
                # Check swap down
                if y < self.GRID_HEIGHT - 1:
                    self._swap_gems((x, y), (x, y+1))
                    if self._check_for_matches_at([(x, y), (x, y+1)]):
                        self._swap_gems((x, y), (x, y+1)) # Swap back
                        return True
                    self._swap_gems((x, y), (x, y+1)) # Swap back
        return False

    def _handle_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[y + empty_count][x] = {'type': self.grid[y][x], 'fall_dist': empty_count}
                    self.grid[y][x] = -1

    def _fill_new_gems(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == -1:
                    self.grid[y][x] = {'type': self.np_random.integers(0, self.NUM_GEM_TYPES), 'fall_dist': self.GRID_HEIGHT}
                elif isinstance(self.grid[y][x], dict):
                    self.grid[y][x] = self.grid[y][x]['type']

    def _add_particles(self, x, y, gem_type):
        px, py = self._grid_to_pixel(x, y)
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.animations.append({
                'type': 'particle', 'pos': (px, py), 'vel': vel,
                'radius': random.uniform(2, 5), 'color': color,
                'progress': 0, 'duration': 30 # Duration here is just lifetime
            })

    def _is_animating(self, anim_type):
        return any(a['type'] == anim_type for a in self.animations)

    def _is_board_clear(self):
        return all(self.grid[y][x] == -1 for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH))

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self.moves_left <= 0 and self.game_state == 'IDLE':
            return True
        if self._is_board_clear():
            self.pending_reward += 100 # Win bonus
            return True
        return False

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Matcher")
    
    running = True
    clock = pygame.time.Clock()
    
    # Game loop
    while running:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_left']}")
        
        if terminated:
            print("Game Over!")
            # Keep showing the final screen for a bit
            for _ in range(90): # 3 seconds
                # We still need to get the observation to render the final screen
                obs, _, _, _, _ = env.step([0,0,0])
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(30)
            running = False
            
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()